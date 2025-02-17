#pragma once

#include <petscksp.h>

#include "mat.h"
#include "util/base.hpp"
#include "util/codi_def.hpp"
#include "vec.h"




AP_NAMESPACE_START

struct ADKSPImpl {
  // Vec vec;

  // Identifier* ad_data;
  // int ad_size;
  // void* transaction_data;

  // FuncCreate createFunc;
  // FuncInit   initFunc;
};

using ADKSP = ADKSPImpl*;

PetscErrorCode KSPCreate                 (MPI_Comm comm, ADKSP *inksp); // TODO: implement
PetscErrorCode KSPDestroy                (ADKSP *ksp); // TODO: implement
PetscErrorCode KSPGetIterationNumber     (ADKSP ksp, PetscInt *its); // TODO: implement
PetscErrorCode KSPGetPC                  (ADKSP ksp, PC *pc); // TODO: implement
PetscErrorCode KSPSetFromOptions         (ADKSP ksp); // TODO: implement
PetscErrorCode KSPSetInitialGuessNonzero (ADKSP ksp, PetscBool flg); // TODO: implement
PetscErrorCode KSPSetOperators           (ADKSP ksp, ADMat Amat, ADMat Pmat); // TODO: implement
PetscErrorCode KSPSolve                  (ADKSP ksp, ADVec b, ADVec x); // TODO: implement
PetscErrorCode KSPView                   (ADKSP ksp, PetscViewer viewer); // TODO: implement

void ADKSPCreateADData(ADKSP ksp);

AP_NAMESPACE_END
