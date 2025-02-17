#pragma once

#include <petscksp.h>

#include "mat.h"
#include "util/base.hpp"
#include "util/codi_def.hpp"
#include "vec.h"

AP_NAMESPACE_START

struct ADKSPImpl {
  KSP ksp;

  // Identifier* ad_data;
  // int ad_size;
  // void* transaction_data;

  // FuncCreate createFunc;
  // FuncInit   initFunc;
};

using ADKSP = ADKSPImpl*;

PetscErrorCode KSPCreate                 (MPI_Comm comm, ADKSP *inksp);
PetscErrorCode KSPDestroy                (ADKSP *ksp);
PetscErrorCode KSPGetIterationNumber     (ADKSP ksp, PetscInt *its);
PetscErrorCode KSPGetPC                  (ADKSP ksp, PC *pc);
PetscErrorCode KSPSetFromOptions         (ADKSP ksp);
PetscErrorCode KSPSetInitialGuessNonzero (ADKSP ksp, PetscBool flg);
PetscErrorCode KSPSetOperators           (ADKSP ksp, ADMat Amat, ADMat Pmat); // TODO: implement
PetscErrorCode KSPSolve                  (ADKSP ksp, ADVec b, ADVec x); // TODO: implement
PetscErrorCode KSPView                   (ADKSP ksp, PetscViewer viewer);

void ADKSPCreateADData(ADKSP ksp);

AP_NAMESPACE_END
