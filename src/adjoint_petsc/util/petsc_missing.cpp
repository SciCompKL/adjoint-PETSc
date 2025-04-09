#include <vector>

#include <adjoint_petsc/util/petsc_missing.h>

#include "mat_iterator_util.hpp"

AP_NAMESPACE_START

/*-------------------------------------------------------------------------------------------------
 * Vec functions
 */

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

/*-------------------------------------------------------------------------------------------------
 * Mat functions
 */

PetscErrorCode MatGetColumnSumAbs(Mat mat, PetscScalar y[]) {
  PetscInt rows;
  PetscInt cols;

  PetscCall(MatGetSize(mat, &rows, &cols));

  std::fill(y, &y[cols], PetscScalar());

  auto func = [&] (PetscInt AP_U(row), PetscInt col, PetscScalar& value) {
    y[col] += abs(value);
  };
  MatIterateAllEntries(func, mat);

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));

  MPI_Allreduce(MPI_IN_PLACE, y, cols, MPI_DOUBLE, MPI_SUM, comm);

  return PETSC_SUCCESS;
}


AP_NAMESPACE_END
