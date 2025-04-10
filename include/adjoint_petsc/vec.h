#pragma once

#include <petscvec.h>

#include "util/base.hpp"
#include "util/codi_def.hpp"
#include "util/wrapper_array.hpp"

AP_NAMESPACE_START

using FuncCreate = std::function<PetscErrorCode(Vec* vec)>;
using FuncInit   = std::function<PetscErrorCode(Vec vec)>;

struct ADVecData {

  int type;

  ADVecData(int type);
  ADVecData(ADVecData const&) = default;

  virtual ~ADVecData() {};
  virtual ADVecData* clone() = 0;

  virtual Identifier* getArray() = 0;
  virtual int         getArraySize() = 0;
  virtual void        restoreArray(Identifier* ids) = 0;


  void checkType(int castType);
};


struct ADVecImpl {
  Vec vec;

  ADVecData* vec_i;

  void* transaction_data;

  FuncCreate createFunc;
  FuncInit   initFunc;
};

using ADVec = ADVecImpl*;

/*-------------------------------------------------------------------------------------------------
 * PETSc functions
 */

PetscErrorCode VecAXPY              (ADVec y, Number alpha, ADVec x);
PetscErrorCode VecAYPX              (ADVec y, Number beta, ADVec x);
PetscErrorCode VecAssemblyBegin     (ADVec vec);
PetscErrorCode VecAssemblyEnd       (ADVec vec);
PetscErrorCode VecCopy              (ADVec x, ADVec y);
PetscErrorCode VecCreate            (MPI_Comm comm, ADVec* vec);
PetscErrorCode VecDestroy           (ADVec* vec);
PetscErrorCode VecDot               (ADVec x, ADVec y, Number *val);
PetscErrorCode VecDuplicate         (ADVec vec, ADVec* newv);
PetscErrorCode VecGetArray          (ADVec vec, WrapperArray* a);
PetscErrorCode VecGetLocalSize      (ADVec vec, PetscInt* size);
PetscErrorCode VecGetOwnershipRange (ADVec x, PetscInt *low, PetscInt *high);
PetscErrorCode VecGetSize           (ADVec vec, PetscInt* size);
PetscErrorCode VecGetValues         (ADVec x, PetscInt ni, PetscInt const* ix, Number* y);
PetscErrorCode VecMax               (ADVec x, PetscInt *p, Number *val);
PetscErrorCode VecNorm              (ADVec x, NormType type, Number *val);
PetscErrorCode VecPointwiseDivide   (ADVec w, ADVec x, ADVec y);
PetscErrorCode VecPointwiseMult     (ADVec w, ADVec x, ADVec y);
PetscErrorCode VecPow               (ADVec v, Number p);
PetscErrorCode VecRestoreArray      (ADVec vec, WrapperArray* a);
PetscErrorCode VecScale             (ADVec x, Number alpha);
PetscErrorCode VecSet               (ADVec x, Number alpha);
PetscErrorCode VecSetFromOptions    (ADVec vec);
PetscErrorCode VecSetOption         (ADVec x, VecOption op, PetscBool flag);
PetscErrorCode VecSetSizes          (ADVec vec, PetscInt m, PetscInt M);
PetscErrorCode VecSetType           (ADVec vec, VecType newType);
PetscErrorCode VecSetValue          (ADVec vec, PetscInt i, Number y, InsertMode iora);
PetscErrorCode VecSetValues         (ADVec vec, PetscInt ni, PetscInt const* ix, Number const* y, InsertMode iora);
PetscErrorCode VecShift             (ADVec v, Number shift);
PetscErrorCode VecSum               (ADVec v, Number *sum);
PetscErrorCode VecView              (ADVec vec, PetscViewer viewer);
PetscErrorCode VecWAXPY             (ADVec w, Number alpha, ADVec x, ADVec y);

/*-------------------------------------------------------------------------------------------------
 * AD specific functions
 */

void ADVecCopyForReverse(ADVec vec, Vec* newv, ADVecData** newd);
void ADVecCreateADData(ADVec vec);
void ADVecIsActive    (ADVec vec, bool* a);

PetscErrorCode ADVecMakePassive(ADVec vec);
PetscErrorCode ADVecRegisterExternalFunctionOutput(ADVec vec);

/*-------------------------------------------------------------------------------------------------
 * Debug functions
 */

void ADVecDebugOutput(ADVec vec, std::string m, int id);
void ADVecDebugOutput(Vec vec, std::string m, int id);

AP_NAMESPACE_END
