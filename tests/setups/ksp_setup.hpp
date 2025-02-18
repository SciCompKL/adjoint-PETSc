#pragma once

#include <gtest/gtest.h>

#include <adjoint_petsc/util/codi_def.hpp>
#include <adjoint_petsc/ksp.h>

#include "mat_setup.hpp"

struct KSPSetup : public MatSetup {
  public:

  using Base = MatSetup;
  static int constexpr ENTRIES_PER_RANK        = Base::ENTRIES_PER_RANK;
  static int constexpr VARIABLE_COUNT          = Base::VARIABLE_COUNT;

  void SetUp() override {
    Base::SetUp();

    PetscCallVoid(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCallVoid(KSPSetFromOptions(ksp));
  }

  void TearDown() override {
    PetscCallVoid(KSPDestroy(&ksp));

    Base::TearDown();
  }

  adjoint_petsc::ADKSP ksp;
};
