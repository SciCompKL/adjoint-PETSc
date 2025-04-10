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
#include <adjoint_petsc/mat.h>

#include "vec_setup.hpp"

struct MatSetup : public VecSetup {
  public:

  using Base = VecSetup;
  static int constexpr ENTRIES_PER_RANK        = Base::ENTRIES_PER_RANK;
  static int constexpr MATRIX_COUNT            = 5;
  static int constexpr MATRIX_NONZERSO_PER_ROW = 3;
  static int constexpr VARIABLE_COUNT          = Base::VARIABLE_COUNT;

  void SetUp() override {
    Base::SetUp();

    for(adjoint_petsc::ADMat& cur : mat) {
      initMat(&cur);
    }
  }

  void TearDown() override {
    for(adjoint_petsc::ADMat& cur : mat) {
      PetscCallVoid(MatDestroy(&cur));
    }

    Base::TearDown();
  }

  void initMat(adjoint_petsc::ADMat* mat) {
    PetscCallVoid(MatCreateAIJ(PETSC_COMM_WORLD, ENTRIES_PER_RANK, ENTRIES_PER_RANK, ENTRIES_PER_RANK* mpi_size, ENTRIES_PER_RANK * mpi_size, MATRIX_NONZERSO_PER_ROW, NULL, MATRIX_NONZERSO_PER_ROW, NULL, mat));
  }

  PetscErrorCode initTriDiagMatrix(adjoint_petsc::ADMat mat, adjoint_petsc::Number* diag, adjoint_petsc::Number* left, adjoint_petsc::Number* right) {
    int low, high;
    PetscCall(MatGetOwnershipRange(mat, &low, &high));

    int matSize = ENTRIES_PER_RANK * mpi_size;

    for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
      PetscCall(MatSetValue(mat, i + low,                 i + low,                 diag[i], INSERT_VALUES));
      PetscCall(MatSetValue(mat, (i + 1 + low) % matSize, i + low,                 left[i], INSERT_VALUES));
      PetscCall(MatSetValue(mat, i + low,                 (i + 1 + low) % matSize, right[i], INSERT_VALUES));
    }

    PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

    return PETSC_SUCCESS;
  }

 std::array<adjoint_petsc::ADMat, MATRIX_COUNT> mat;
};
