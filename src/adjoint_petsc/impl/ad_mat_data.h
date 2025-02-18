#pragma once

#include <adjoint_petsc/util/base.hpp>
#include <adjoint_petsc/util/codi_def.hpp>
#include <adjoint_petsc/mat.h>

AP_NAMESPACE_START

enum class ADMatType {
  NONE = 0,
  MatAIJ,
  MatSeqAIJ
};


struct ADMatSeqAIJData : public ADMatData {
  static int constexpr TYPE = (int)ADMatType::MatSeqAIJ;

  std::vector<Identifier> index;

  ADMatSeqAIJData(ADMatSeqAIJData const&) = default;
  explicit ADMatSeqAIJData(PetscInt size);

  ADMatSeqAIJData* clone() override;

  static ADMatSeqAIJData* cast(ADMatData* d);

};

struct ADMatAIJData : public ADMatData {
  static int constexpr TYPE = (int)ADMatType::MatAIJ;
  ADMatSeqAIJData index_d;
  ADMatSeqAIJData index_o;

  ADMatAIJData(ADMatAIJData const&) = default;
  ADMatAIJData(PetscInt diag_size, PetscInt off_diag_size);

  ADMatAIJData* clone() override;

  static ADMatAIJData* cast(ADMatData* d);
};

AP_NAMESPACE_END
