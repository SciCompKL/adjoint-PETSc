#include "ad_mat_data.h"

AP_NAMESPACE_START

ADMatData::ADMatData(int type) : type(type) {}

ADMatSeqAIJData::ADMatSeqAIJData(PetscInt size) :
    ADMatData(TYPE),
    index(size) {}

ADMatSeqAIJData* ADMatSeqAIJData::clone() {
  return new ADMatSeqAIJData(*this);
}

ADMatSeqAIJData* ADMatSeqAIJData::cast(ADMatData* d) {
  if(d->type != TYPE) {
    // TODO: Throw error
  }

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
  if(d->type != TYPE) {
    // TODO: Throw error
  }

  return dynamic_cast<ADMatAIJData*>(d);
}

AP_NAMESPACE_END
