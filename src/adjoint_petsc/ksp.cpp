#include "../../include/adjoint_petsc/ksp.h"

AP_NAMESPACE_START

PetscErrorCode KSPCreate(MPI_Comm comm, ADKSP *inksp) {
  *inksp = new ADKSPImpl();

  PetscCall(KSPCreate(comm, &(*inksp)->ksp));

  return PETSC_SUCCESS;
}

PetscErrorCode KSPDestroy                (ADKSP *ksp) {
  PetscCall(KSPDestroy(&(*ksp)->ksp));

  delete *ksp;

  return PETSC_SUCCESS;
}

PetscErrorCode KSPGetIterationNumber(ADKSP ksp, PetscInt *its) {
  return KSPGetIterationNumber(ksp->ksp, its);
}

PetscErrorCode KSPGetPC(ADKSP ksp, PC *pc) {
  return KSPGetPC(ksp->ksp, pc);
}

PetscErrorCode KSPSetFromOptions(ADKSP ksp) {
  return KSPSetFromOptions(ksp->ksp);
}

PetscErrorCode KSPSetInitialGuessNonzero(ADKSP ksp, PetscBool flg) {
  return KSPSetInitialGuessNonzero(ksp->ksp, flg);
}

// PetscErrorCode KSPSetOperators           (ADKSP ksp, ADMat Amat, ADMat Pmat); // TODO: implement
// PetscErrorCode KSPSolve                  (ADKSP ksp, ADVec b, ADVec x); // TODO: implement

PetscErrorCode KSPView(ADKSP ksp, PetscViewer viewer) {
  // TODO: Implement AD version
  return KSPView(ksp->ksp, viewer);
}

AP_NAMESPACE_END
