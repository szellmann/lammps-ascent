#include <assert.h>
#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>
#include <mpi.h>

#define LAMMPS_LIB_MPI 1
#include <lammps.h>
#include <domain.h>
#include <library.h>
#include <modify.h>
#include <fix.h>
#include <fix_external.h>

#include <ascent.hpp>
#include <conduit.hpp>

using namespace LAMMPS_NS;

struct Info {
  int rank;
  LAMMPS *lmp;
};

void mycallback(void *ptr, bigint ntimestep,
        int nlocal, int *id, double **x, double **f)
{
  Info *info = (Info *)ptr;
  double **pos = (double **)lammps_extract_atom(info->lmp, "x");

  conduit::Node mesh;
  mesh["coordsets/coords/type"] = "explicit";
  mesh["coordsets/coords/values/x"].set(conduit::DataType::float64(nlocal));
  mesh["coordsets/coords/values/y"].set(conduit::DataType::float64(nlocal));
  mesh["coordsets/coords/values/z"].set(conduit::DataType::float64(nlocal));

  conduit::float64 *xvals = mesh["coordsets/coords/values/x"].value();
  conduit::float64 *yvals = mesh["coordsets/coords/values/y"].value();
  conduit::float64 *zvals = mesh["coordsets/coords/values/z"].value();

  for (int i=0;i<nlocal;++i) {
    double x = (*pos)[i*3];
    double y = (*pos)[i*3+1];
    double z = (*pos)[i*3+2];
    //std::cout << x << ',' << y << ',' << z << '\n';
    xvals[i] = x;
    yvals[i] = y;
    zvals[i] = z;
  }

  mesh["topologies/mesh/type"] = "unstructured";
  mesh["topologies/mesh/coordset"] = "coords";
  mesh["topologies/mesh/elements/shape"] = "point";

  mesh["topologies/mesh/elements/connectivity"].set(conduit::DataType::int32(nlocal));

  conduit::int32 *conn = mesh["topologies/mesh/elements/connectivity"].value();
  for(int i=0; i<nlocal;++i) {
    conn[i] = i;
  }

  // TODO: var1 -> this var!
  mesh["fields/var1/association"] = "vertex";
  mesh["fields/var1/topology"] = "mesh";
  mesh["fields/var1/values"].set(conduit::DataType::float64(nlocal));

  conduit::float64 *vals = mesh["fields/var1/values"].value();
  for (int i=0;i<nlocal;++i) {
    vals[i] = rand()/double(RAND_MAX);
  }

  conduit::Node verify_info;
  if (!conduit::blueprint::mesh::verify(mesh,verify_info)) {
    std::cerr << verify_info.to_yaml() << '\n';
    exit(1);
  }
  // else {
  //   std::cout << "Verification passed!\n";
  //   std::cout << mesh.to_yaml() << '\n';
  // }

  ascent::Ascent a;
  a.open();
  a.publish(mesh);

  conduit::Node actions;
  conduit::Node &add_act = actions.append();
  add_act["action"] = "add_scenes";

  conduit::Node &scenes = add_act["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "var1";
  scenes["s1/plots/p1/points/radius"] = 1.f;
  scenes["s1/image_prefix"] = "out_ascent_render_points";

  std::cout << actions.to_yaml() << '\n';
  a.execute(actions);
  a.close();
}

static void usage() {
  std::cout <<
    "Usage: ./lammps-ascent -in <lammps input> [options] [-lmp <lammps args>]\n"
    "Options:\n"
    "  -h                  Print this help text\n"
    "  -lmp <lammps args>  Pass the list of arguments <args> to LAMMPS as if they were\n"
    "                      command line args to LAMMPS. This must be the last argument, all\n"
    "                      following arguments will be passed to lammps.\n";
  exit(1);
}

int main(int argc, char *argv[])
{
  std::string inFileName = "";
  std::vector<char *> lammps_args(1,argv[0]);

  for (int i=1;i<argc;++i) {
    const std::string arg = argv[i];
    if (arg[0] != '-') {
      usage();
    }
    else if (arg == "-h" || arg == "-help") {
      usage();
    }
    else if (arg == "-i" || arg == "-in") {
      inFileName = argv[++i];
    }
    else if (arg == "-lmp") {
      ++i;
      for (;i<argc;++i) {
        lammps_args.push_back(argv[i]);
      }
    }
  }

  if (inFileName.empty())
    usage();

  MPI_Init(&argc,&argv);
  MPI_Comm lammps_comm = MPI_COMM_WORLD;
  int comm_rank, comm_size;
  MPI_Comm_rank(lammps_comm,&comm_rank);
  MPI_Comm_size(lammps_comm,&comm_size);

  LAMMPS *lammps;
  #if LAMMPS_LIB_MPI
  lammps_open(lammps_args.size(), lammps_args.data(), lammps_comm, (void **)&lammps);
  #else
  assert(comm_size==1);
  lammps_open_no_mpi(lammps_args.size(), lammps_args.data(), (void **)&lammps);
  #endif

  // process lammps in-file line by line
  if (comm_rank == 0) {
    std::ifstream in(inFileName.c_str());
    for (std::string line; std::getline(in,line);) {
      if (line.empty())
        continue;
      int len = line.size();
      MPI_Bcast(&len, 1, MPI_INT, 0, lammps_comm);
      MPI_Bcast((void *)line.c_str(), len, MPI_CHAR, 0, lammps_comm);
      lammps_command(lammps, line.c_str());
    }
    // Bcast out we're done with the file
    int len = 0;
    MPI_Bcast(&len, 1, MPI_INT, 0, lammps_comm);
  } else {
    for (;;) {
      int len = 0;
      MPI_Bcast(&len, 1, MPI_INT, 0, lammps_comm);
      if (len == 0)
        break;
      std::vector<char> line(len+1,'\0');
      MPI_Bcast(line.data(), len, MPI_CHAR, 0, lammps_comm);
      lammps_command(lammps, line.data());
    }
  }

  // Setup the fix external callback
  Info info;
  info.rank = comm_rank;
  info.lmp = lammps;

  lammps_command(lammps, "fix myfix all external pf/callback 1 1");
  lammps_set_fix_external_callback(lammps, "myfix", mycallback, &info);
  lammps_command(lammps, "run 10000");

  lammps_close(lammps);

  MPI_Finalize();
}


