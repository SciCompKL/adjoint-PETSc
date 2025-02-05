#include <vector>

#include "../../../include/adjoint_petsc/util/petsc_missing.h"



AP_NAMESPACE_START

PetscErrorCode VecGetValuesNonLocal(Vec vec, PetscInt ni, PetscInt ix[], PetscScalar y[]) {
  Vec         temp;
  VecScatter  scatter;
  IS          from, to;
  PetscScalar *values;

  // Create local ids
  std::vector<PetscInt> idx_to(ni);
  for(PetscInt i = 0; i < ni; i += 1) { idx_to[i] = i; }

  PetscCall(VecCreateSeq(PETSC_COMM_SELF,ni,&temp));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,ni,ix,PETSC_USE_POINTER,&from));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,ni,idx_to.data(),PETSC_USE_POINTER,&to));
  PetscCall(VecScatterCreate(vec,from,temp,to,&scatter));
  PetscCall(VecScatterBegin(scatter,vec,temp,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,vec,temp,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecGetArray(temp,&values));

  for(PetscInt i = 0; i < ni; i += 1) {
    y[i] = values[i];
  }

  PetscCall(VecRestoreArray(temp, &values));

  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&temp));

  return PETSC_SUCCESS;
}

AP_NAMESPACE_END
