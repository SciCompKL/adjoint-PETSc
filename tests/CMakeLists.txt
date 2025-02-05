cmake_minimum_required(VERSION 3.20)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(adjoint_petsc_tests
    main_test.cpp
    test_vec.cpp)

target_link_libraries(adjoint_petsc_tests PUBLIC gtest adjoint_petsc)

set_property(TARGET adjoint_petsc_tests PROPERTY CROSSCOMPILING_EMULATOR '${PETSc_DIR}/bin/mpirun -n 2')
gtest_discover_tests(adjoint_petsc_tests)

