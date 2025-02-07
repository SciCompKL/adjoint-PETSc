#include "../../include/adjoint_petsc/mat.h"

AP_NAMESPACE_START

// PetscErrorCode MatAssemblyBegin          (ADMat mat);
// PetscErrorCode MatAssemblyEnd            (ADMat mat);
// PetscErrorCode MatConvert                (ADMat mat, MatType newtype, MatReuse reuse, ADMat *M);
// PetscErrorCode MatCreate                 (MPI_Comm comm, ADMat* mat);
// PetscErrorCode MatCreateAIJ              (MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], ADMat *A);
// PetscErrorCode MatCreateDense            (MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Number *data, ADMat *A);
// PetscErrorCode MatDestroy                (ADMat* mat);
// PetscErrorCode MatDuplicate              (ADMat mat, ADMat* newv);

PetscErrorCode MatGetInfo(ADMat mat, MatInfoType flag, MatInfo *info) {
  return MatGetInfo(mat->mat, flag, info);
}

PetscErrorCode MatGetLocalSize(ADMat mat, PetscInt *m, PetscInt *n) {
  return MatGetLocalSize(mat->mat, m, n);
}

PetscErrorCode MatGetOwnershipRange(ADMat mat, PetscInt *m, PetscInt *n) {
  return MatGetOwnershipRange(mat->mat, m, n);
}

PetscErrorCode MatGetSize(ADMat mat, PetscInt *m, PetscInt *n) {
  return MatGetSize(mat->mat, m, n);
}

PetscErrorCode MatGetType                (ADMat mat, MatType *type) {
  return MatGetType(mat->mat, type);
}

// PetscErrorCode MatGetValues              (ADMat mat, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], Number v[]);
// PetscErrorCode MatMPIAIJSetPreallocation (ADMat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[]);
// PetscErrorCode MatMPIAIJSetPreallocation (ADMat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[]);
// PetscErrorCode MatMult                   (ADMat mat, Vec x, Vec y);
// PetscErrorCode MatNorm                   (ADMat x, NormType type, Number *val);
// PetscErrorCode MatSeqAIJSetPreallocation (ADMat B, PetscInt nz, const PetscInt nnz[]);
PetscErrorCode MatSetFromOptions(ADMat mat) {
  return MatSetFromOptions(mat->mat);
}

PetscErrorCode MatSetOption(ADMat x, MatOption op, PetscBool flag) {
  return MatSetOption(x->mat, op, flag);
}

PetscErrorCode MatSetSizes(ADMat mat, PetscInt m, PetscInt n, PetscInt M, PetscInt N) {
  PetscCall(MatSetSizes(mat->mat, m, n, M, N));

  ADMatCreateADData(mat);
  return PETSC_SUCCESS;

}

// PetscErrorCode MatSetValues              (ADMat mat, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], const Number v[], InsertMode addv);
PetscErrorCode MatView(ADMat mat, PetscViewer viewer) {
  return MatView(mat->mat, viewer);
}

// PetscErrorCode MatZeroEntries            (ADMat mat);

AP_NAMESPACE_END
