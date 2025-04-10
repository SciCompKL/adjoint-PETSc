/*
 * adjoint-PETSc
 *
 * Copyright (C) 2025 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
 * Homepage: http://scicomp.rptu.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
 *
 * This file is part of adjoint-PETSc (GITHUB_LINK).
 *
 * adjoint-PETSc is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * adjoint-PETSc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License for more details.
 * You should have received a copy of the GNU
 * Lesser General Public License along with adjoint-PETSc.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
 */

#pragma once

#include <gtest/gtest.h>

#include <adjoint_petsc/util/codi_def.hpp>
#include <adjoint_petsc/vec.h>

struct VecSetup : public testing::Test {
  public:

  static int constexpr ENTRIES_PER_RANK = 4;
  static int constexpr VECTOR_COUNT     = 5;
  static int constexpr VARIABLE_COUNT   = 5;

  void SetUp() override {
    MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);

    mpi_rank_next = (mpi_rank + 1) % mpi_size;
    mpi_rank_prev = (mpi_rank + mpi_size - 1) % mpi_size;

    tape = &adjoint_petsc::Number::getTape();
    tape->reset();
    tape->gradient(0) = -42.0; // Set an invalid value for the zero gradient.
    tape->setActive();

    for(adjoint_petsc::ADVec& cur : vec) {
      initVec(&cur);
    }

    int i = 1;
    for(adjoint_petsc::Number& cur: s) {
      cur = i + mpi_rank * 10;
      tape->registerInput(cur);

      i += 1;
    }
  }

  void TearDown() override {
    for(adjoint_petsc::ADVec& cur : vec) {
      PetscCallVoid(VecDestroy(&cur));
    }

    tape->setPassive();
  }

  void initVec(adjoint_petsc::ADVec* vec) {
    PetscCallVoid(VecCreate(PETSC_COMM_WORLD, vec));
    PetscCallVoid(VecSetSizes(*vec, ENTRIES_PER_RANK, ENTRIES_PER_RANK * mpi_size));
    PetscCallVoid(VecSetFromOptions(*vec));
  }

  PetscErrorCode setVector(adjoint_petsc::ADVec vec, int size, adjoint_petsc::Number* s) {
    adjoint_petsc::WrapperArray values = {};
    PetscCall(VecGetArray(vec, &values));
    for(int i = 0; i < size; i += 1) { values[i] = s[i]; }
    PetscCall(VecRestoreArray(vec, &values));

    return PETSC_SUCCESS;
  }

  void evaluateTape() {
      EXPECT_DOUBLE_EQ(tape->gradient(0), -42.0);
      tape->evaluate();
      EXPECT_DOUBLE_EQ(tape->gradient(0), -42.0);
  }

  std::array<adjoint_petsc::ADVec, VECTOR_COUNT> vec;

  std::array<adjoint_petsc::Number, VARIABLE_COUNT> s;

  int mpi_rank;
  int mpi_rank_next;
  int mpi_rank_prev;
  int mpi_size;

  adjoint_petsc::Tape* tape;

};
