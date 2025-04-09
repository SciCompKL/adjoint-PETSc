#include "ad_mat_data.h"
#include "../util/exception.hpp"

AP_NAMESPACE_START

ADMatData::ADMatData(int type) : type(type) {}

void ADMatData::checkType(int castType) {
  if(type != castType) {
    AP_EXCEPTION("Cast to wrong type %d. Correct type is %d", type, castType);
  }
}

ADMatSeqAIJData::ADMatSeqAIJData(PetscInt size) :
    ADMatData(TYPE),
    index(size) {}

ADMatSeqAIJData* ADMatSeqAIJData::clone() {
  return new ADMatSeqAIJData(*this);
}

ADMatSeqAIJData* ADMatSeqAIJData::cast(ADMatData* d) {
  d->checkType(TYPE);

  return dynamic_cast<ADMatSeqAIJData*>(d);
}


ADMatAIJData::ADMatAIJData(PetscInt diag_size, PetscInt off_diag_size) :
    ADMatData(TYPE),
    index_d(diag_size),
    index_o(off_diag_size) {}

ADMatAIJData* ADMatAIJData::clone() {
  return new ADMatAIJData(*this);
}

ADMatAIJData* ADMatAIJData::cast(ADMatData* d) {
  d->checkType(TYPE);

  return dynamic_cast<ADMatAIJData*>(d);
}

ADMatType ADMatDataPTypeToEnum(MatType ptype) {
  if(0 == strcmp(ptype, "mpiaij")) {
    return ADMatType::MatAIJ;
  }
  else if(0 == strcmp(ptype, "seqaij")) {
    return ADMatType::MatAIJ;
  }
  else {
    return ADMatType::NONE;
  }
}

ADMatType ADMatGetADType(ADMat mat) {
  if(nullptr != mat->mat_i) {
    return (ADMatType)mat->mat_i->type;
  } else {
    return ADMatGetADType(mat->mat);
  }
}

ADMatType ADMatGetADType(Mat mat) {
  MatType ptype;
  MatGetType(mat, &ptype);

  return ADMatDataPTypeToEnum(ptype);
}

AP_NAMESPACE_END
