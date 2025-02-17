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

TEST_F(VecSetup, VecAXPY) {

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

TEST_F(VecSetup, VecAYPX) {

  PetscCallVoid(VecSet(vec[0], s[0]));
  PetscCallVoid(VecSet(vec[1], s[1]));

  PetscCallVoid(VecAYPX(vec[1], s[2], vec[0]));

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[1], &values));

  adjoint_petsc::Real target = s[1].getValue() * s[2].getValue() + s[0].getValue();
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    auto temp = values[i];
    EXPECT_EQ(target, temp.getValue());
    tape->registerOutput(temp);
    temp.setGradient(i + 100 * mpi_rank);
  }
  PetscCallVoid(VecRestoreArray(vec[1], &values));

  tape->evaluate();

  EXPECT_EQ(s[0].getGradient(), 1 + 2 + 3 +  ENTRIES_PER_RANK * 100 * mpi_rank);
  EXPECT_EQ(s[1].getGradient(), s[2].getValue() * (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank));
  EXPECT_EQ(s[2].getGradient(), s[1].getValue() * (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank));
}

TEST_F(VecSetup, VecShift) {

  PetscCallVoid(VecSet(vec[0], s[0]));

  PetscCallVoid(VecShift(vec[0], s[2]));

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[0], &values));

  adjoint_petsc::Real target = s[2].getValue() + s[0].getValue();
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    auto temp = values[i];
    EXPECT_EQ(target, temp.getValue());
    tape->registerOutput(temp);
    temp.setGradient(i + 100 * mpi_rank);
  }
  PetscCallVoid(VecRestoreArray(vec[0], &values));

  tape->evaluate();

  EXPECT_EQ(s[0].getGradient(), 1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank);
  EXPECT_EQ(s[2].getGradient(), 1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank);
}

TEST_F(VecSetup, VecMax_PosNo) {

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[0], &values));
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) { values[i] = s[0]; }
  PetscCallVoid(VecRestoreArray(vec[0], &values));

  adjoint_petsc::Number v = {};
  PetscCallVoid(VecMax(vec[0], nullptr, &v));


  EXPECT_EQ(v.getValue(), (mpi_size - 1) * 10.0 + 1.0);
  tape->registerOutput(v);
  v.setGradient(pow(10.0, mpi_rank));

  tape->evaluate();

  adjoint_petsc::Real target_b = 0.0;
  for(int i = 0; i < mpi_size; i += 1) {
    target_b += pow(10.0, i);
  }

  if(mpi_rank + 1 == mpi_size) {
    EXPECT_EQ(s[0].getGradient(), ENTRIES_PER_RANK * target_b); // All four vector entries get the value
  }
  else {
    EXPECT_EQ(s[0].getGradient(), 0); // No maximum value.
  }
}

TEST_F(VecSetup, VecMax_PosYes) {

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[0], &values));
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) { values[i] = s[0]; }
  PetscCallVoid(VecRestoreArray(vec[0], &values));

  PetscInt pos;
  adjoint_petsc::Number v = {};
  PetscCallVoid(VecMax(vec[0], &pos, &v));


  EXPECT_EQ(v.getValue(), (mpi_size - 1) * 10.0 + 1.0);
  EXPECT_EQ(pos, (mpi_size - 1) * ENTRIES_PER_RANK);
  tape->registerOutput(v);
  v.setGradient(pow(10.0, mpi_rank));

  tape->evaluate();

  adjoint_petsc::Real target_b = 0.0;
  for(int i = 0; i < mpi_size; i += 1) {
    target_b += pow(10.0, i);
  }

  if(mpi_rank + 1 == mpi_size) {
    EXPECT_EQ(s[0].getGradient(),  target_b); // Only the first entry is set.
  }
  else {
    EXPECT_EQ(s[0].getGradient(), 0); // No maximum value.
  }
}

TEST_F(VecSetup, VecNorm_1) {

  std::array<adjoint_petsc::Number, VARIABLE_COUNT> init = s;
  adjoint_petsc::Number v = {};
  init[0] = -init[0];
  PetscCallVoid(setVector(vec[0], ENTRIES_PER_RANK, init.data()));
  PetscCallVoid(VecNorm(vec[0], NORM_1, &v));

  EXPECT_EQ(v.getValue(), (1 + 2 + 3 + 4) * mpi_size + (mpi_size - 1) * mpi_size * 10.0 * ENTRIES_PER_RANK / 2.0);
  tape->registerOutput(v);
  v.setGradient(pow(10.0, mpi_rank));

  tape->evaluate();

  adjoint_petsc::Real target_b = 0.0;
  for(int i = 0; i < mpi_size; i += 1) {
    target_b += pow(10.0, i);
  }

  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    EXPECT_EQ(s[i].getGradient(),  target_b);
  }
}

TEST_F(VecSetup, VecNorm_2) {

  std::array<adjoint_petsc::Number, VARIABLE_COUNT> init = s;
  adjoint_petsc::Number v = {};
  init[0] = -init[0];
  PetscCallVoid(setVector(vec[0], ENTRIES_PER_RANK, init.data()));
  PetscCallVoid(VecNorm(vec[0], NORM_2, &v));

  adjoint_petsc::Real target = 0.0;
  for(int i = 0; i < mpi_size; i += 1) {
    for(int d = 0; d < ENTRIES_PER_RANK; d += 1) {
      adjoint_petsc::Real t = d + 1 + 10 * i;
      target += t * t;
    }
  }
  EXPECT_DOUBLE_EQ(v.getValue(), sqrt(target));
  tape->registerOutput(v);
  v.setGradient(pow(10.0, mpi_rank));

  tape->evaluate();

  adjoint_petsc::Real target_b = 0.0;
  for(int i = 0; i < mpi_size; i += 1) {
    target_b += pow(10.0, i);
  }

  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    EXPECT_EQ(s[i].getGradient(),  2 * s[i].getValue() * target_b);
  }
}

TEST_F(VecSetup, VecNorm_MAX) {

  std::array<adjoint_petsc::Number, VARIABLE_COUNT> init = s;
  adjoint_petsc::Number v = {};
  init[0] = -init[0];
  PetscCallVoid(setVector(vec[0], ENTRIES_PER_RANK, init.data()));
  PetscCallVoid(VecNorm(vec[0], NORM_MAX, &v));

  EXPECT_DOUBLE_EQ(v.getValue(), 4.0  + (mpi_size - 1) * 10.0);
  tape->registerOutput(v);
  v.setGradient(pow(10.0, mpi_rank));

  tape->evaluate();

  adjoint_petsc::Real target_b = 0.0;
  for(int i = 0; i < mpi_size; i += 1) {
    target_b += pow(10.0, i);
  }

  for(int i = 0; i < ENTRIES_PER_RANK - 1; i += 1) {
    EXPECT_EQ(s[i].getGradient(),  0);
  }
  if(mpi_rank + 1 == mpi_size) {
    EXPECT_EQ(s[3].getGradient(), target_b);
  }
  else {
    EXPECT_EQ(s[3].getGradient(),  0);
  }
}

TEST_F(VecSetup, VecNorm_1_AND_2) {

  std::array<adjoint_petsc::Number, VARIABLE_COUNT> init = s;
  std::array<adjoint_petsc::Number, 2> v = {};
  init[0] = -init[0];
  PetscCallVoid(setVector(vec[0], ENTRIES_PER_RANK, init.data()));
  PetscCallVoid(VecNorm(vec[0], NORM_1_AND_2, v.data()));

  adjoint_petsc::Real target = 0.0;
  for(int i = 0; i < mpi_size; i += 1) {
    for(int d = 0; d < ENTRIES_PER_RANK; d += 1) {
      adjoint_petsc::Real t = d + 1 + 10 * i;
      target += t * t;
    }
  }
  EXPECT_DOUBLE_EQ(v[0].getValue(), (1 + 2 + 3 + 4) * mpi_size + (mpi_size - 1) * mpi_size * 10.0 * ENTRIES_PER_RANK / 2.0);
  EXPECT_DOUBLE_EQ(v[1].getValue(), sqrt(target));
  tape->registerOutput(v[0]);
  tape->registerOutput(v[1]);
  v[0].setGradient(pow(10.0, mpi_rank));
  v[1].setGradient(2.0 * pow(10.0, mpi_rank));

  tape->evaluate();

  adjoint_petsc::Real target_0_b = 0.0;
  adjoint_petsc::Real target_1_b = 0.0;
  for(int i = 0; i < mpi_size; i += 1) {
    target_0_b += pow(10.0, i);
    target_1_b += 2.0 * pow(10.0, i);
  }

  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    EXPECT_EQ(s[i].getGradient(),  2 * s[i].getValue() * target_1_b + target_0_b);
  }
}

TEST_F(VecSetup, VecScale) {

  PetscCallVoid(VecSet(vec[0], s[0]));

  PetscCallVoid(VecScale(vec[0], s[2]));

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[0], &values));

  adjoint_petsc::Real target = s[2].getValue() * s[0].getValue();
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    auto temp = values[i];
    EXPECT_EQ(target, temp.getValue());
    tape->registerOutput(temp);
    temp.setGradient(i + 100 * mpi_rank);
  }
  PetscCallVoid(VecRestoreArray(vec[0], &values));

  tape->evaluate();

  EXPECT_EQ(s[0].getGradient(), s[2].getValue() * (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank));
  EXPECT_EQ(s[2].getGradient(), s[0].getValue() * (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank));
}

TEST_F(VecSetup, VecPow) {

  PetscCallVoid(VecSet(vec[0], s[0]));

  PetscCallVoid(VecPow(vec[0], s[2]));

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[0], &values));

  adjoint_petsc::Real target = pow(s[0].getValue(), s[2].getValue());
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    auto temp = values[i];
    EXPECT_EQ(target, temp.getValue());
    tape->registerOutput(temp);
    temp.setGradient(i + 100 * mpi_rank);
  }
  PetscCallVoid(VecRestoreArray(vec[0], &values));

  tape->evaluate();

  EXPECT_EQ(s[0].getGradient(), s[2].getValue() * pow(s[0], (s[2].getValue() - 1.0)) * (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank));
  EXPECT_EQ(s[2].getGradient(), log(s[0].getValue()) * target * (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank));
}

TEST_F(VecSetup, VecSum) {

  std::array<adjoint_petsc::Number, VARIABLE_COUNT> init = s;
  adjoint_petsc::Number v = {};
  init[0] = -init[0];
  PetscCallVoid(setVector(vec[0], ENTRIES_PER_RANK, init.data()));
  PetscCallVoid(VecSum(vec[0], &v));

  EXPECT_EQ(v.getValue(), (-1 + 2 + 3 + 4) * mpi_size + (mpi_size - 1) * mpi_size * 10.0 * (ENTRIES_PER_RANK - 2) / 2.0);
  tape->registerOutput(v);
  v.setGradient(pow(10.0, mpi_rank));

  tape->evaluate();

  adjoint_petsc::Real target_b = 0.0;
  for(int i = 0; i < mpi_size; i += 1) {
    target_b += pow(10.0, i);
  }

  EXPECT_EQ(s[0].getGradient(),  -target_b);
  for(int i = 1; i < ENTRIES_PER_RANK; i += 1) {
    EXPECT_EQ(s[i].getGradient(),  target_b);
  }
}

TEST_F(VecSetup, VecDot) {

  PetscCallVoid(VecSet(vec[0], s[0]));
  PetscCallVoid(VecSet(vec[1], s[1]));

  adjoint_petsc::Number v = 0.0;
  PetscCallVoid(VecDot(vec[0], vec[1], &v));

  adjoint_petsc::Real target = 0.0;
  for(int i = 0; i < mpi_size; i += 1) {
    target += (1.0 + 10 * i) * (2.0 + 10.0 * i);
  }
  EXPECT_EQ(ENTRIES_PER_RANK * target, v.getValue());
  tape->registerOutput(v);
  v.setGradient(100);

  tape->evaluate();

  EXPECT_EQ(s[0].getGradient(), mpi_size * ENTRIES_PER_RANK * s[1].getValue() * 100);
  EXPECT_EQ(s[1].getGradient(), mpi_size * ENTRIES_PER_RANK * s[0].getValue() * 100);
}

TEST_F(VecSetup, VecPointwiseDevide) {

  PetscCallVoid(VecSet(vec[0], s[0]));
  PetscCallVoid(VecSet(vec[1], s[1]));

  PetscCallVoid(VecPointwiseDivide(vec[2], vec[0], vec[1]));

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[2], &values));

  adjoint_petsc::Real target = s[0].getValue() / s[1].getValue();
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    auto temp = values[i];
    EXPECT_EQ(target, temp.getValue());
    tape->registerOutput(temp);
    temp.setGradient(i + 100 * mpi_rank);
  }
  PetscCallVoid(VecRestoreArray(vec[2], &values));

  tape->evaluate();

  EXPECT_DOUBLE_EQ(s[0].getGradient(), (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank) / s[1].getValue());
  EXPECT_DOUBLE_EQ(s[1].getGradient(), -(1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank) * target / s[1].getValue());
}

TEST_F(VecSetup, VecPointwiseMult) {

  PetscCallVoid(VecSet(vec[0], s[0]));
  PetscCallVoid(VecSet(vec[1], s[1]));

  PetscCallVoid(VecPointwiseMult(vec[2], vec[0], vec[1]));

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[2], &values));

  adjoint_petsc::Real target = s[0].getValue() * s[1].getValue();
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    auto temp = values[i];
    EXPECT_EQ(target, temp.getValue());
    tape->registerOutput(temp);
    temp.setGradient(i + 100 * mpi_rank);
  }
  PetscCallVoid(VecRestoreArray(vec[2], &values));

  tape->evaluate();

  EXPECT_EQ(s[0].getGradient(), (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank) * s[1].getValue());
  EXPECT_EQ(s[1].getGradient(), (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank) * s[0].getValue());
}

TEST_F(VecSetup, VecWAXPY) {

  PetscCallVoid(VecSet(vec[0], s[0]));
  PetscCallVoid(VecSet(vec[1], s[1]));

  PetscCallVoid(VecWAXPY(vec[2], s[2], vec[0], vec[1]));

  adjoint_petsc::WrapperArray values = {};
  PetscCallVoid(VecGetArray(vec[2], &values));

  adjoint_petsc::Real target = s[0].getValue() * s[2].getValue() + s[1].getValue();
  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    auto temp = values[i];
    EXPECT_EQ(target, temp.getValue());
    tape->registerOutput(temp);
    temp.setGradient(i + 100 * mpi_rank);
  }
  PetscCallVoid(VecRestoreArray(vec[2], &values));

  tape->evaluate();

  EXPECT_EQ(s[1].getGradient(), 1 + 2 + 3 +  ENTRIES_PER_RANK * 100 * mpi_rank);
  EXPECT_EQ(s[0].getGradient(), s[2].getValue() * (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank));
  EXPECT_EQ(s[2].getGradient(), s[0].getValue() * (1 + 2 + 3 + ENTRIES_PER_RANK * 100 * mpi_rank));
}
