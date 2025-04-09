
#include <algorithm>

#include "addata_helper.hpp"
#include "vec_iterator_util.hpp"

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

  PetscCall(getAdjointNoReset(vec_b, vi, dim));

  // Reset afterwards since ids can contain duplicates.
  for(size_t i = 0; i < ids.size(); i += 1) {
    vi->resetAdjoint(ids[i], dim);
  }

  return PETSC_SUCCESS;
}

PetscErrorCode AdjointVecData::getAdjointNoReset(Vec vec_b, VectorInterface* vi, PetscInt dim) {
  auto func = [&](PetscInt AP_U(row), Real& value, Identifier id) {
    value = vi->getAdjoint(id, dim);
  };
  PetscCall(VecIterateAllEntries(func, vec_b, ids.data()));

  return PETSC_SUCCESS;
}

PetscErrorCode AdjointVecData::updateAdjoint(Vec vec_b, VectorInterface* vi, PetscInt dim) {
  Tape& tape = Number::getTape();

  auto func = [&](PetscInt AP_U(row), Real& value, Identifier id) {
    if( tape.isIdentifierActive(id)) {
        vi->updateAdjoint(id, dim, value);
    }
  };
  PetscCall(VecIterateAllEntries(func, vec_b, ids.data()));

  return PETSC_SUCCESS;
}

AP_NAMESPACE_END
