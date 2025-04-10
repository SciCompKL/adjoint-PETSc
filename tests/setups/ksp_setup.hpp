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
