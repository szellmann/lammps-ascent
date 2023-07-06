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
#include <ascent_vtkh_data_adapter.hpp>
#include <conduit.hpp>

#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayCopyDevice.h>
#include <vtkm/cont/ArrayHandleRandomUniformReal.h>
#include <vtkm/cont/DataSetBuilderExplicit.h>
#include <vtkm/filter/density_estimate/ParticleDensityCloudInCell.h>
#include <vtkm/filter/density_estimate/ParticleDensityNearestGridPoint.h>
#include <vtkm/io/VTKDataSetWriter.h>

using namespace LAMMPS_NS;

struct Info {
  MPI_Comm comm;
  int rank;
  LAMMPS *lmp;
};

void mycallback(void *ptr, bigint ntimestep,
        int nlocal, int *id, double **x, double **f)
{
  Info *info = (Info *)ptr;
  double **pos = (double **)lammps_extract_atom(info->lmp, "x");

  
  double xsublo = info->lmp->domain->sublo[0];
  double xsubhi = info->lmp->domain->subhi[0];
  double ysublo = info->lmp->domain->sublo[1];
  double ysubhi = info->lmp->domain->subhi[1];
  double zsublo = info->lmp->domain->sublo[2];
  double zsubhi = info->lmp->domain->subhi[2];

#if 1
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

#if 0
  auto mesh_vtkm = ascent::VTKHDataAdapter::BlueprintToVTKmDataSet(mesh, /* attempt to zero-copy = */true, "mesh");
  mesh_vtkm->PrintSummary(std::cout);
  std::stringstream str;
  str << "particles-t" << ntimestep << '_' << info->rank << ".vtk";
  vtkm::io::VTKDataSetWriter writer(str.str());
  writer.WriteDataSet(*mesh_vtkm);
#endif

  conduit::Node ascent_opts;
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(info->comm);

  ascent::Ascent a;
  a.open(ascent_opts);
  a.publish(mesh);

  conduit::Node actions;

  conduit::Node &add_act = actions.append();
  add_act["action"] = "add_scenes";

  conduit::Node &scenes = add_act["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "var1";
  scenes["s1/image_prefix"] = "out_ascent_render_points";

  // render as points
  scenes["s1/plots/p1/points/radius"] = .3f;
#endif

#if 0
  std::vector<vtkm::Vec3f> hPos(nlocal);
  for (int i=0;i<nlocal;++i) {
    double x = (*pos)[i*3];
    double y = (*pos)[i*3+1];
    double z = (*pos)[i*3+2];
    hPos[i] = {x,y,z};
  }

  vtkm::cont::ArrayHandle<vtkm::Vec3f> positions
    = vtkm::cont::make_ArrayHandle(hPos.data(), nlocal, vtkm::CopyFlag::Off);

  vtkm::cont::ArrayHandle<vtkm::Id> connectivity;
  vtkm::cont::ArrayCopy(vtkm::cont::make_ArrayHandleIndex(nlocal), connectivity);

  auto dataSet = vtkm::cont::DataSetBuilderExplicit::Create(
    positions, vtkm::CellShapeTagVertex{}, 1, connectivity);

  std::vector<vtkm::FloatDefault> hVar1(nlocal);
  for (int i=0;i<nlocal;++i) {
    hVar1[i] = rand()/double(RAND_MAX);
  }
  vtkm::cont::ArrayHandle<vtkm::FloatDefault> var1
    = vtkm::cont::make_ArrayHandle(hVar1.data(), nlocal, vtkm::CopyFlag::Off);
  dataSet.AddCellField("var1", var1);

  vtkm::Id3 cellDims = { 64, 64, 64 };
  vtkm::Bounds bounds = { { xsublo, xsubhi }, { ysublo, ysubhi }, { zsublo, zsubhi } };
  vtkm::filter::density_estimate::ParticleDensityNearestGridPoint filter;
  filter.SetDimension(cellDims);
  filter.SetBounds(bounds);
  filter.SetActiveField("var1");

  auto density = filter.Execute(dataSet);

  vtkm::cont::ArrayHandle<vtkm::FloatDefault> field;
  density.GetCellField("density").GetData().AsArrayHandle<vtkm::FloatDefault>(field);

  std::stringstream str;
  str << "test-t" << ntimestep << '_' << info->rank << ".vtk";
  vtkm::io::VTKDataSetWriter writer(str.str());
  writer.WriteDataSet(density);

  conduit::Node vtkmBP;
  ascent::VTKHDataAdapter::VTKmToBlueprintDataSet(&density, vtkmBP, "mesh", /* attempt to zero-copy = */true);
  vtkmBP.print();

  conduit::Node verify_info;
  if (!conduit::blueprint::mesh::verify(vtkmBP,verify_info)) {
    std::cerr << verify_info.to_yaml() << '\n';
    exit(1);
  }
  // else {
  //   std::cout << "Verification passed!\n";
  //   std::cout << mesh.to_yaml() << '\n';
  // }

  conduit::Node ascent_opts;
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(info->comm);

  ascent::Ascent a;
  a.open(ascent_opts);
  a.publish(vtkmBP);

  conduit::Node actions;

  conduit::Node &add_pl = actions.append();
  add_pl["action"] = "add_pipelines";

  conduit::Node &pipelines = add_pl["pipelines"];

  pipelines["p1/f1/type"] = "contour";
  pipelines["p1/f1/params/field"] = "density";
  //pipelines["p1/f1/params/levels"] = "15";
  pipelines["p1/f1/params/iso_values"] = 0.5;

  conduit::Node &add_act = actions.append();
  add_act["action"] = "add_scenes";

  conduit::Node &scenes = add_act["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "density";
  scenes["s1/image_prefix"] = "out_ascent_render_contour";

  // render iso contours
  scenes["s1/plots/p1/pipeline"] = "p1";
#endif

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
  info.comm = lammps_comm;
  info.rank = comm_rank;
  info.lmp = lammps;

  lammps_command(lammps, "fix myfix all external pf/callback 1 1");
  lammps_set_fix_external_callback(lammps, "myfix", mycallback, &info);
  lammps_command(lammps, "run 10000");

  lammps_close(lammps);

  MPI_Finalize();
}


