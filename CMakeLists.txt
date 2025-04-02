cmake_minimum_required(VERSION 3.20)
project(adjoint_petsc
  VERSION 0.1.0
  LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CoDiPack_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/codipack/cmake CACHE PATH "Path to CoDiPack cmake files.")
set(PETSc_DIR     Not_Defined CACHE PATH "Path to petsc")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

find_package(CoDiPack CONFIG REQUIRED)

add_library(adjoint_petsc
  src/adjoint_petsc/ksp.cpp
  src/adjoint_petsc/mat.cpp
  src/adjoint_petsc/options.cpp
  src/adjoint_petsc/util/addata_helper.cpp
  src/adjoint_petsc/util/petsc_missing.cpp
  src/adjoint_petsc/vec.cpp
  src/adjoint_petsc/impl/ad_mat_data.cpp
)

target_sources(adjoint_petsc
  PUBLIC
    FILE_SET api
    TYPE HEADERS
    BASE_DIRS include
    FILES
      include/adjoint_petsc/ksp.h
      include/adjoint_petsc/mat.h
      include/adjoint_petsc/options.h
      include/adjoint_petsc/vec.h
      include/adjoint_petsc/util/base.hpp
      include/adjoint_petsc/util/codi_def.hpp
      include/adjoint_petsc/util/wrapper_array.hpp
)

target_include_directories(adjoint_petsc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${PETSc_DIR}/include)
target_compile_options    (adjoint_petsc PRIVATE -Wall -pedantic)

target_link_libraries(adjoint_petsc PUBLIC CoDiPack)
target_link_libraries(adjoint_petsc INTERFACE petsc mpi)
target_link_directories(adjoint_petsc INTERFACE ${PETSc_DIR}/lib)

install(TARGETS adjoint_petsc FILE_SET api)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
