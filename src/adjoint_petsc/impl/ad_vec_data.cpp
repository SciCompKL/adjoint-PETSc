#include "../util/exception.hpp"
#include "ad_vec_data.h"

AP_NAMESPACE_START

ADVecData::ADVecData(int type) : type(type) {}

ADVecLocalData::ADVecLocalData(PetscInt size) : ADVecData(TYPE), index(size) {}

void ADVecData::checkType(int castType) {
  if(type != castType) {
    AP_EXCEPTION("Cast to wrong type %d. Correct type is %d", type, castType);
  }
}

ADVecLocalData* ADVecLocalData::clone() {
  return new ADVecLocalData(*this);
}

ADVecLocalData* ADVecLocalData::cast(ADVecData* d) {
  d->checkType(TYPE);

  return dynamic_cast<ADVecLocalData*>(d);
}

Identifier* ADVecLocalData::getArray() {
  return index.data();
}

int ADVecLocalData::getArraySize() {
  return index.size();
}

void ADVecLocalData::restoreArray(Identifier* AP_U(ids)) {
  // Do nothing
}

ADVecType ADVecDataPTypeToEnum(VecType ptype) {
  if(0 == strcmp(ptype, "mpi")) {
    return ADVecType::VecMPI;
  }
  else {
    return ADVecType::NONE;
  }
}

ADVecType ADVecGetADType(ADVec vec) {
  if(nullptr != vec->vec_i) {
    return (ADVecType)vec->vec_i->type;
  } else {
    return ADVecGetADType(vec->vec);
  }
}

ADVecType ADVecGetADType(Vec vec) {
  VecType ptype;
  VecGetType(vec, &ptype);

  return ADVecDataPTypeToEnum(ptype);
}


AP_NAMESPACE_END
