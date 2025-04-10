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

#include <memory>

#include <petscksp.h>

#include "mat.h"
#include "util/base.hpp"
#include "util/codi_def.hpp"
#include "vec.h"

AP_NAMESPACE_START

using SharedKSP = std::shared_ptr<KSP>;

struct ADKSPImpl {
  // TODO: Current implementation retains the created ksp for the reverse evaluation.
  //       Implement a version/option that a separate one is create. Then PC needs to
  //       be wrapped so that the changes to the PC can be tracked.
  SharedKSP ksp;

  ADMat Amat;
  ADMat Pmat;

  KSP& getKSP();
};

using ADKSP = ADKSPImpl*;

/*-------------------------------------------------------------------------------------------------
 * PETSc functions
 */

PetscErrorCode KSPCreate                 (MPI_Comm comm, ADKSP *inksp);
PetscErrorCode KSPDestroy                (ADKSP *ksp);
PetscErrorCode KSPGetIterationNumber     (ADKSP ksp, PetscInt *its);
PetscErrorCode KSPGetPC                  (ADKSP ksp, PC *pc);
PetscErrorCode KSPSetFromOptions         (ADKSP ksp);
PetscErrorCode KSPSetInitialGuessNonzero (ADKSP ksp, PetscBool flg);
PetscErrorCode KSPSetOperators           (ADKSP ksp, ADMat Amat, ADMat Pmat);
PetscErrorCode KSPSolve                  (ADKSP ksp, ADVec b, ADVec x);
PetscErrorCode KSPView                   (ADKSP ksp, PetscViewer viewer);

AP_NAMESPACE_END
