cmake_minimum_required(VERSION 3.20)
project(adjoint_petsc
  VERSION 0.1.0
  LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CoDiPack_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/codipack/cmake CACHE PATH "Path to CoDiPack cmake files.")
set(PESTc_DIR     Not_Defined CACHE PATH "Path to petsc")

find_package(CoDiPack CONFIG REQUIRED)

add_library(adjoint_petsc
  src/adjoint_petsc/vec.cpp
  src/adjoint_petsc/mat.cpp
  src/adjoint_petsc/util/petsc_missing.cpp
  src/adjoint_petsc/util/addata_helper.cpp
)

target_include_directories(adjoint_petsc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${PETSc_DIR}/include)

target_link_libraries(adjoint_petsc PUBLIC CoDiPack)
target_link_libraries(adjoint_petsc INTERFACE petsc mpi)
target_link_directories(adjoint_petsc INTERFACE ${PETSc_DIR}/lib)

add_subdirectory(tests)
