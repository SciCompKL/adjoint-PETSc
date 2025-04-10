/*
 * adjoint-PETSc
 *
 * Copyright (C) 2025 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
 * Homepage: http://scicomp.rptu.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
 *
 * This file is part of adjoint-PETSc (GITHUB_LINK).
 *
 * adjoint-PETSc is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * adjoint-PETSc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License for more details.
 * You should have received a copy of the GNU
 * Lesser General Public License along with adjoint-PETSc.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
 */

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
