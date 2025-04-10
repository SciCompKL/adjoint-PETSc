#pragma once

#include <adjoint_petsc/util/base.hpp>
#include <adjoint_petsc/util/codi_def.hpp>
#include <adjoint_petsc/vec.h>

AP_NAMESPACE_START

enum class ADVecType {
  NONE = 0,
  VecMPI
};

struct ADVecLocalData : public ADVecData {
  static int constexpr TYPE = (int)ADVecType::VecMPI;

  std::vector<Identifier> index;

  ADVecLocalData(ADVecLocalData const&) = default;
  ADVecLocalData(PetscInt size);

  ADVecLocalData* clone() override;

  Identifier* getArray() override;
  int         getArraySize()  override;
  void        restoreArray(Identifier* ids) override;

  static ADVecLocalData* cast(ADVecData* d);
};

ADVecType ADVecDataPTypeToEnum(VecType ptype);

ADVecType ADVecGetADType(ADVec vec);
ADVecType ADVecGetADType(Vec vec);

AP_NAMESPACE_END
