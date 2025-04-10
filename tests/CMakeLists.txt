#
# adjoint-PETSc
#
# Copyright (C) 2025 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
# Homepage: http://scicomp.rptu.de
# Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
#
# Lead developers: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
#
# This file is part of adjoint-PETSc (GITHUB_LINK).
#
# adjoint-PETSc is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# adjoint-PETSc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU
# Lesser General Public License along with adjoint-PETSc.
# If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
#

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
    test_ksp.cpp
    test_mat.cpp
    test_vec.cpp
  )

target_link_libraries(adjoint_petsc_tests PUBLIC gtest adjoint_petsc)
target_include_directories(adjoint_petsc_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set_property(TARGET adjoint_petsc_tests PROPERTY CROSSCOMPILING_EMULATOR '${PETSc_DIR}/bin/mpirun -n 2')
gtest_discover_tests(adjoint_petsc_tests)

