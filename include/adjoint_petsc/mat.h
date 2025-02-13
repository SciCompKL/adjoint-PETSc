#pragma once

#include <petscmat.h>

#include "util/base.hpp"
#include "util/codi_def.hpp"

AP_NAMESPACE_START

using FuncMatCreate = std::function<PetscErrorCode(Mat* mat)>;
using FuncMatInit   = std::function<PetscErrorCode(Mat mat)>;

struct ADMatImpl {
  Mat mat;

  Identifier* ad_data;
  int ad_size;
  int ad_size_diag;
  void* transaction_data;

  FuncMatCreate createFunc;
  FuncMatInit   initFunc;
};

using ADMat = ADMatImpl*;

PetscErrorCode MatAssemblyBegin          (ADMat mat, MatAssemblyType type);
PetscErrorCode MatAssemblyEnd            (ADMat mat, MatAssemblyType type);
// PetscErrorCode MatConvert                (ADMat mat, MatType newtype, MatReuse reuse, ADMat *M);
PetscErrorCode MatCreate                 (MPI_Comm comm, ADMat* mat);
PetscErrorCode MatCreateAIJ              (MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], ADMat *A);
// PetscErrorCode MatCreateDense            (MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Number *data, ADMat *A);
PetscErrorCode MatDestroy                (ADMat* mat);
PetscErrorCode MatDuplicate              (ADMat mat, MatDuplicateOption op, ADMat* newv);
PetscErrorCode MatGetInfo                (ADMat mat, MatInfoType flag, MatInfo *info);
PetscErrorCode MatGetLocalSize           (ADMat mat, PetscInt *m, PetscInt *n);
PetscErrorCode MatGetOwnershipRange      (ADMat mat, PetscInt *m, PetscInt *n);
PetscErrorCode MatGetSize                (ADMat mat, PetscInt *m, PetscInt *n);
PetscErrorCode MatGetType                (ADMat mat, MatType *type);
PetscErrorCode MatGetValue               (ADMat mat, PetscInt row, PetscInt col, Number* v);
PetscErrorCode MatGetValues              (ADMat mat, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], Number v[]);
// PetscErrorCode MatMPIAIJSetPreallocation (ADMat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[]);
// PetscErrorCode MatMult                   (ADMat mat, Vec x, Vec y);
// PetscErrorCode MatNorm                   (ADMat x, NormType type, Number *val);
// PetscErrorCode MatSeqAIJSetPreallocation (ADMat B, PetscInt nz, const PetscInt nnz[]);
PetscErrorCode MatSetFromOptions         (ADMat mat);
PetscErrorCode MatSetOption              (ADMat x, MatOption op, PetscBool flag);
PetscErrorCode MatSetSizes               (ADMat mat, PetscInt m, PetscInt n, PetscInt M, PetscInt N);
PetscErrorCode MatSetValues              (ADMat mat, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], const Number v[], InsertMode addv);
PetscErrorCode MatSetValue               (ADMat mat, PetscInt i, PetscInt j, Number v, InsertMode addv);
PetscErrorCode MatView                   (ADMat mat, PetscViewer viewer);
// PetscErrorCode MatZeroEntries            (ADMat mat);

void ADMatCreateADData(ADMat mat);

AP_NAMESPACE_END
