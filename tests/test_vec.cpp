#include <gtest/gtest.h>

#include "setups/vec_setup.hpp"


TEST_F(VecSetup, VecSetValues) {

  // Always set 0 entry on first rank.
  // Set the 1 entry on this rank.
  // Set the 2 entry on the next rank.
  std::array<PetscInt, 3> pos = {0, mpi_rank * ENTRIES_PER_RANK + 1, mpi_rank_next * ENTRIES_PER_RANK + 2};
  PetscCallVoid(VecSetValues(vec[0], 3, pos.data(), s.data(), ADD_VALUES));
  PetscCallVoid(VecAssemblyBegin(vec[0]));
  PetscCallVoid(VecAssemblyEnd(vec[0]));

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[0], &values));

  if(0 == mpi_rank) {
    int sum = mpi_size * (mpi_size - 1) / 2;
    EXPECT_EQ(values[0].getValue(), mpi_size + 10 * sum);
  }
  else {
    EXPECT_EQ(values[0].getValue(), 0.0);
  }

  EXPECT_EQ(values[1].getValue(), 2 + mpi_rank * 10);
  EXPECT_EQ(values[2].getValue(), 3 + mpi_rank_prev * 10);

  for(int i = 0; i < 3; i += 1) {
    adjoint_petsc::Wrapper w = values[i];
    tape->registerOutput(w);
    w.setGradient(100 + 10 * mpi_rank + i + 1);
  }

  PetscCallVoid(VecRestoreArray(vec[0], &values));

  tape->evaluate();

  EXPECT_EQ(s[0].getGradient(), 101.0); // Adjoint from first rank.
  EXPECT_EQ(s[1].getGradient(), 102.0 + 10 * mpi_rank); // Adjoint from own rank.
  EXPECT_EQ(s[2].getGradient(), 103.0 + 10 * mpi_rank_next); // Adjoint from next rank.
}

TEST_F(VecSetup, VecSetValuesSteps) {

  // Always set 0 entry on first rank.
  // Set the 1 entry on this rank.
  // Set the 2 entry on the next rank.
  std::array<PetscInt, 3> pos = {0, mpi_rank * ENTRIES_PER_RANK + 1, mpi_rank_next * ENTRIES_PER_RANK + 2};
  PetscCallVoid(VecSetValues(vec[0], 1, &pos[0], &s[0], ADD_VALUES));
  PetscCallVoid(VecSetValues(vec[0], 1, &pos[1], &s[1], ADD_VALUES));
  PetscCallVoid(VecSetValues(vec[0], 1, &pos[2], &s[2], ADD_VALUES));
  PetscCallVoid(VecAssemblyBegin(vec[0]));
  PetscCallVoid(VecAssemblyEnd(vec[0]));

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[0], &values));

  if(0 == mpi_rank) {
    int sum = mpi_size * (mpi_size - 1) / 2;
    EXPECT_EQ(values[0].getValue(), mpi_size + 10 * sum);
  }
  else {
    EXPECT_EQ(values[0].getValue(), 0.0);
  }

  EXPECT_EQ(values[1].getValue(), 2 + mpi_rank * 10);
  EXPECT_EQ(values[2].getValue(), 3 + mpi_rank_prev * 10);

  for(int i = 0; i < 3; i += 1) {
    adjoint_petsc::Wrapper w = values[i];
    tape->registerOutput(w);
    w.setGradient(100 + 10 * mpi_rank + i + 1);
  }

  PetscCallVoid(VecRestoreArray(vec[0], &values));

  tape->evaluate();

  EXPECT_EQ(s[0].getGradient(), 101.0); // Adjoint from first rank.
  EXPECT_EQ(s[1].getGradient(), 102.0 + 10 * mpi_rank); // Adjoint from own rank.
  EXPECT_EQ(s[2].getGradient(), 103.0 + 10 * mpi_rank_next); // Adjoint from next rank.
}

TEST_F(VecSetup, VecCopy) {

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[0], &values));
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) { values[i] = s[i]; }
  PetscCallVoid(VecRestoreArray(vec[0], &values));

  PetscCallVoid(VecCopy(vec[0], vec[1]));

  PetscCallVoid(VecGetArray(vec[1], &values));
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    auto temp = values[i];
    EXPECT_EQ(s[i].getValue(), temp.getValue());
    tape->registerOutput(temp);
    temp.setGradient(i + 100 * mpi_rank);
  }
  PetscCallVoid(VecRestoreArray(vec[1], &values));

  tape->evaluate();

  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    EXPECT_EQ(s[i].getGradient(), i + 100 * mpi_rank);
  }
}

TEST_F(VecSetup, VecGetValues) {

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[0], &values));
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) { values[i] = s[i]; }
  PetscCallVoid(VecRestoreArray(vec[0], &values));

  std::array<PetscInt, 2> ix = {0, 3};
  std::array<adjoint_petsc::Number, 2> v = {};
  PetscCallVoid(VecGetValues(vec[0], 2, ix.data(), v.data()));

  EXPECT_EQ(v[0].getValue(), s[0].getValue());
  EXPECT_EQ(v[1].getValue(), s[3].getValue());

  for(int i = 0; i < 2; i += 1) {
    tape->registerOutput(v[i]);
    v[i].setGradient(i + 100 * mpi_rank);
  }

  tape->evaluate();

  EXPECT_EQ(s[0].getGradient(), 0 + 100 * mpi_rank);
  EXPECT_EQ(s[1].getGradient(), 0);
  EXPECT_EQ(s[2].getGradient(), 0);
  EXPECT_EQ(s[3].getGradient(), 1 + 100 * mpi_rank);
}

TEST_F(VecSetup, VecSet) {

  PetscCallVoid(VecSet(vec[0], s[0]));

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[0], &values));

  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    auto temp = values[i];
    EXPECT_EQ(s[0].getValue(), temp.getValue());
    tape->registerOutput(temp);
    temp.setGradient(i + 100 * mpi_rank);
  }
  PetscCallVoid(VecRestoreArray(vec[0], &values));

  tape->evaluate();

  EXPECT_EQ(s[0].getGradient(), 1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank);
}

TEST_F(VecSetup, VecAXPYt) {

  PetscCallVoid(VecSet(vec[0], s[0]));
  PetscCallVoid(VecSet(vec[1], s[1]));

  PetscCallVoid(VecAXPY(vec[1], s[2], vec[0]));

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[1], &values));

  adjoint_petsc::Real target = s[0].getValue() * s[2].getValue() + s[1].getValue();
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    auto temp = values[i];
    EXPECT_EQ(target, temp.getValue());
    tape->registerOutput(temp);
    temp.setGradient(i + 100 * mpi_rank);
  }
  PetscCallVoid(VecRestoreArray(vec[1], &values));

  tape->evaluate();

  EXPECT_EQ(s[1].getGradient(), 1 + 2 + 3 +  ENTRIES_PER_RANK * 100 * mpi_rank);
  EXPECT_EQ(s[0].getGradient(), s[2].getValue() * (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank));
  EXPECT_EQ(s[2].getGradient(), s[0].getValue() * (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank));
}
