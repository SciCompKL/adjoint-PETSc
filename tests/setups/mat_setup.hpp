#pragma once

#include <gtest/gtest.h>

#include <adjoint_petsc/util/codi_def.hpp>
#include <adjoint_petsc/mat.h>

#include "vec_setup.hpp"

struct MatSetup : public VecSetup {
  protected:

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
