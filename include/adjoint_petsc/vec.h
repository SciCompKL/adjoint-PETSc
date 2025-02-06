#pragma once

#include <petscvec.h>

#include "util/base.hpp"
#include "util/codi_def.hpp"
#include "util/wrapper_array.hpp"

AP_NAMESPACE_START

using FuncCreate = std::function<PetscErrorCode(Vec* vec)>;
using FuncInit   = std::function<PetscErrorCode(Vec vec)>;

struct ADVecImpl {
  Vec vec;

  Identifier* ad_data;
  int ad_size;
  void* transaction_data;

  FuncCreate createFunc;
  FuncInit   initFunc;
};

using ADVec = ADVecImpl*;

PetscErrorCode VecCreate        (MPI_Comm comm, ADVec* vec);
PetscErrorCode VecSetFromOptions(ADVec vec);
PetscErrorCode VecSetSizes      (ADVec vec, PetscInt m, PetscInt M);
PetscErrorCode VecSetType       (ADVec vec, VecType newType);
PetscErrorCode VecSetValues     (ADVec vec, PetscInt ni, PetscInt const* ix, Number const* y, InsertMode iora);
PetscErrorCode VecSetValue      (ADVec vec, PetscInt i, Number y, InsertMode iora);
PetscErrorCode VecAssemblyBegin (ADVec vec);
PetscErrorCode VecAssemblyEnd   (ADVec vec);
PetscErrorCode VecDestroy       (ADVec* vec);
PetscErrorCode VecGetArray      (ADVec vec, WrapperArray* a);
PetscErrorCode VecRestoreArray  (ADVec vec, WrapperArray* a);
PetscErrorCode VecDuplicate     (ADVec vec, ADVec* newv);
PetscErrorCode VecCopy          (ADVec x, ADVec y);
PetscErrorCode VecView          (ADVec vec, PetscViewer viewer);
PetscErrorCode VecGetSize       (ADVec vec, PetscInt* size);
PetscErrorCode VecGetLocalSize  (ADVec vec, PetscInt* size);
PetscErrorCode VecGetValues     (ADVec x, PetscInt ni, PetscInt const* ix, Number* y);
PetscErrorCode VecSet           (ADVec x, Number alpha);
PetscErrorCode VecAXPY          (ADVec y, Number alpha, ADVec x);
PetscErrorCode VecAYPX          (ADVec y, Number beta, ADVec x);
PetscErrorCode VecShift         (ADVec v, Number shift);



void ADVecCreateADData(ADVec vec);


AP_NAMESPACE_END
