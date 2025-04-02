#include <gtest/gtest.h>

#include <adjoint_petsc/util/mat_iterator_util.hpp>

#include "setups/ksp_setup.hpp"


TEST_F(KSPSetup, Solve) {
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> diag;
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> left;
  std::array<adjoint_petsc::Number, ENTRIES_PER_RANK> right;
  diag.fill(s[0]);
  left.fill(s[1]);
  right.fill(s[2]);
  PetscCallVoid(initTriDiagMatrix(mat[0], diag.data(), left.data(), right.data()));

  PetscCallVoid(VecSet(vec[0], s[3]));

  PetscCallVoid(KSPSetOperators(ksp, mat[0], mat[0]));

  PetscCallVoid(KSPSolve(ksp, vec[0], vec[1]));

  std::array<adjoint_petsc::Real, ENTRIES_PER_RANK * 2> expected_values = {
     6.0511086506945433e-01,
    -7.7736021148746892e-02,
     9.5583809700327770e-01,
     1.0665446484314043e+00,
     3.4059305252068017e-01,
     6.2464516349305155e-01,
     2.3398358240986528e-01,
     3.0234143319806489e-01,
  };
  // TODO: Throw error if mpi_ranks != 2

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[1], &values));
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    EXPECT_DOUBLE_EQ(values[i].getValue(), expected_values[i + mpi_rank * ENTRIES_PER_RANK]);

    values[i].setGradient(100 + 10 * mpi_rank);
  }
  PetscCallVoid(VecRestoreArray(vec[1], &values));

  evaluateTape();

  std::array<adjoint_petsc::Real, 2> expected_gradient_s0 = {
    -31.671653651023323e+00,
    -5.7056884773163032e+00
  };
  std::array<adjoint_petsc::Real, 2> expected_gradient_s1 = {
    -16.894665079987327,
    -4.3014379189128462
  };
  std::array<adjoint_petsc::Real, 2> expected_gradient_s2 = {
    -53.629178677011353,
    -6.1091820226503879
  };
  std::array<adjoint_petsc::Real, 2> expected_gradient_s4 = {
      59.508325854533922,
      13.008172213984674
  };

  EXPECT_DOUBLE_EQ(s[0].getGradient(), expected_gradient_s0[mpi_rank]);
  EXPECT_DOUBLE_EQ(s[1].getGradient(), expected_gradient_s1[mpi_rank]);
  EXPECT_DOUBLE_EQ(s[2].getGradient(), expected_gradient_s2[mpi_rank]);
  EXPECT_DOUBLE_EQ(s[3].getGradient(), expected_gradient_s4[mpi_rank]);
}
