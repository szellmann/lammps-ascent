project(lammps-ascent LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.10)
set(LAMMPS_DIR "/home/szellmann/lammps")
set(LAMMPS_INCLUDE_DIRS ${LAMMPS_INCLUDE_DIRS} ${LAMMPS_DIR}/src)
set(LAMMPS_LIBRARIES ${LAMMPS_LIBRARIES} "${LAMMPS_DIR}/build/liblammps.so")

find_package(Ascent)
find_package(CUDAToolkit)
find_package(MPI COMPONENTS C CXX REQUIRED)
find_package(VTKm REQUIRED QUIET)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
  PRIVATE
  lammps-ascent.cu
)
target_include_directories(${PROJECT_NAME} PRIVATE ${LAMMPS_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_libraries(${PROJECT_NAME} PUBLIC ${LAMMPS_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ascent::ascent_mpi)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::toolkit)
target_link_libraries(${PROJECT_NAME} PUBLIC 
  ${MPI_C_LIBRARIES}
  ${MPI_C_LINK_FLAGS}
  ${MPI_CXX_LIBRARIES}
  ${MPI_CXX_LINK_FLAGS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
  vtkm::io
  vtkm::filter_entity_extraction
  vtkm::filter_density_estimate
)

vtkm_add_target_information(${PROJECT_NAME}
  DROP_UNUSED_SYMBOLS
  MODIFY_CUDA_FLAGS
  DEVICE_SOURCES lammps-ascent.cu
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_VISIBILITY_PRESET "hidden")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_VISIBILITY_PRESET "hidden")
