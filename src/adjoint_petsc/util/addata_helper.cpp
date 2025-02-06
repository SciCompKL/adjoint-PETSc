#include "../../../include/adjoint_petsc/util/addata_helper.hpp"

#include <algorithm>

AP_NAMESPACE_START

AdjointVecData::AdjointVecData(ADVec vec) : ids(vec->ad_size), global_size(0), createFunc(vec->createFunc), initFunc(vec->initFunc) {
  std::copy(vec->ad_data, &vec->ad_data[vec->ad_size], ids.begin());
  PetscCallVoid(VecGetSize(vec->vec, &global_size));
}


PetscErrorCode AdjointVecData::createAdjoint(Vec* vec_b, PetscInt dimSize) {
  PetscCall(createFunc(vec_b));
  PetscCall(initFunc(*vec_b));
  PetscCall(VecSetSizes(*vec_b, ids.size() * dimSize, global_size * dimSize));

  return PETSC_SUCCESS;
}

PetscErrorCode AdjointVecData::freeAdjoint(Vec* vec_b) {
  PetscCall(VecDestroy(vec_b));

  return PETSC_SUCCESS;
}

PetscErrorCode AdjointVecData::getAdjoint(Vec vec_b, VectorInterface* vi, PetscInt dim) {
  Real* adjoint;
  PetscCall(VecGetArray(vec_b, &adjoint));
  for(size_t i = 0; i < ids.size(); i += 1) {
    adjoint[i] = vi->getAdjoint(ids[i], dim);
    vi->resetAdjoint(ids[i], dim);
  }

  PetscCall(VecRestoreArray(vec_b, &adjoint));

  return PETSC_SUCCESS;
}

PetscErrorCode AdjointVecData::registerExternalFunctionOutput(ADVec vec) {
  Tape& tape = Number::getTape();
  Real* primals;

  PetscCall(VecGetArray(vec->vec, &primals));
  for(PetscInt i = 0; i < vec->ad_size; i += 1) {
    Wrapper temp = createRefType(primals[i], vec->ad_data[i]);
    tape.registerExternalFunctionOutput(temp);
  }

  PetscCall(VecRestoreArray(vec->vec, &primals));

  return PETSC_SUCCESS;
}

AP_NAMESPACE_END
