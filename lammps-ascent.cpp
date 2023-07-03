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


using namespace LAMMPS_NS;

struct Info {
  int rank;
  LAMMPS *lmp;
};

void mycallback(void *ptr, bigint ntimestep,
        int nlocal, int *id, double **x, double **f)
{
  std::cout << nlocal << '\n';
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


