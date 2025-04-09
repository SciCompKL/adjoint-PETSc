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
