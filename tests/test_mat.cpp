#include <gtest/gtest.h>

#include <adjoint_petsc/util/mat_iterator_util.hpp>

#include "setups/mat_setup.hpp"


TEST_F(MatSetup, SetValues) {


  std::array<PetscInt, 4> rows = {0, mpi_rank * ENTRIES_PER_RANK + 1, mpi_rank * ENTRIES_PER_RANK + 2,      mpi_rank_next * ENTRIES_PER_RANK + 3};
  std::array<PetscInt, 4> cols = {0, mpi_rank * ENTRIES_PER_RANK + 1, mpi_rank_next * ENTRIES_PER_RANK + 2, mpi_rank * ENTRIES_PER_RANK + 3};
  std::array<adjoint_petsc::Number, 4> values = {s[0], s[1], s[2], s[3]};

  for(int i = 0;i < 4; i += 1) {
    PetscCallVoid(MatSetValues(mat[0], 1, &rows[i], 1, &cols[i], &values[i], ADD_VALUES));
  }

  PetscCallVoid(MatAssemblyBegin(mat[0], MAT_FINAL_ASSEMBLY));
  PetscCallVoid(MatAssemblyEnd(mat[0], MAT_FINAL_ASSEMBLY));


  auto func = [&](PetscInt row, PetscInt col, adjoint_petsc::Wrapper& value) {
    tape->registerOutput(value);

    if(0 == row && col == 0) {
      EXPECT_EQ(value.getValue(), 12.0);
      value.setGradient(100 + 10 * mpi_rank);
    } else if(1 == row % ENTRIES_PER_RANK) {
      EXPECT_EQ(value.getValue(), mpi_rank * 10 + 2);
      value.setGradient(1000 + 10 * mpi_rank);
    } else if(2 == row % ENTRIES_PER_RANK) {
      EXPECT_EQ(value.getValue(), mpi_rank * 10 + 3);
      value.setGradient(10000 + 10 * mpi_rank);
    } else if(3 == row % ENTRIES_PER_RANK) {
      EXPECT_EQ(value.getValue(), mpi_rank_prev * 10 + 4);
      value.setGradient(100000 + 10 * mpi_rank);
    } else {
      // TODO: Throw error.
    }
  };
  adjoint_petsc::ADObjIterateAllEntries(mat[0], func);

  tape->evaluate();

  EXPECT_EQ(s[0].getGradient(), 100.0); // Adjoint from first rank.
  EXPECT_EQ(s[1].getGradient(), 1000.0 + 10 * mpi_rank); // Adjoint from own rank.
  EXPECT_EQ(s[2].getGradient(), 10000.0 + 10 * mpi_rank); // Adjoint from own rank.
  EXPECT_EQ(s[3].getGradient(), 100000.0 + 10 * mpi_rank_next); // Adjoint from next rank.
}
