#pragma once

#include <gtest/gtest.h>

#include <adjoint_petsc/util/codi_def.hpp>
#include <adjoint_petsc/vec.h>

struct VecSetup : public testing::Test {
  protected:

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
    PetscCallVoid(VecSetFromOptions(*vec));
    PetscCallVoid(VecSetSizes(*vec, ENTRIES_PER_RANK, ENTRIES_PER_RANK * mpi_size));
  }

  std::array<adjoint_petsc::ADVec, VECTOR_COUNT> vec;

  std::array<adjoint_petsc::Number, VARIABLE_COUNT> s;

  int mpi_rank;
  int mpi_rank_next;
  int mpi_rank_prev;
  int mpi_size;

  adjoint_petsc::Tape* tape;

};
