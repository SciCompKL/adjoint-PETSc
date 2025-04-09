#pragma once

#include <adjoint_petsc/util/base.hpp>
#include <adjoint_petsc/util/codi_def.hpp>
#include <adjoint_petsc/vec.h>

AP_NAMESPACE_START

template<typename Impl>
struct ReverseDataBase {

  /*-------------------------------------------------------------------------------------------------
   * Interface functions
   */

  void reverse(Tape* tape, VectorInterface* vi);

  /*-------------------------------------------------------------------------------------------------
   * Helper functions
   */

  void push() {
    Number::getTape().pushExternalFunction(codi::ExternalFunction<Tape>::create(&reverse, this, &free));
  }

  static void reverse(Tape* tape, void* d, VectorInterface* vi) {
    Impl* data = (Impl*)d;

    data->reverse(tape, vi);
  }

  static void free(Tape* tape, void* d) {
    (void) tape;

    Impl* data = (Impl*)d;

    delete data;
  }
};

struct AdjointVecData {
  std::vector<Identifier> ids;

  PetscInt   global_size;
  FuncCreate createFunc;
  FuncInit   initFunc;

  AdjointVecData(ADVec vec);

  PetscErrorCode createAdjoint(Vec* vec_b, PetscInt dimSize);
  PetscErrorCode freeAdjoint(Vec* vec_b);
  PetscErrorCode getAdjoint(Vec vec_b, VectorInterface* vi, PetscInt dim);
  PetscErrorCode getAdjointNoReset(Vec vec_b, VectorInterface* vi, PetscInt dim);
  PetscErrorCode updateAdjoint(Vec vec_b, VectorInterface* vi, PetscInt dim);
};

AP_NAMESPACE_END
