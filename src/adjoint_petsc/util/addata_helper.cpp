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


#include <algorithm>

#include "addata_helper.hpp"
#include "vec_iterator_util.hpp"

AP_NAMESPACE_START

AdjointVecData::AdjointVecData(ADVec vec) : ids(vec->vec_i->clone()), global_size(0), createFunc(vec->createFunc), initFunc(vec->initFunc) {
  PetscCallVoid(VecGetSize(vec->vec, &global_size));
}

AdjointVecData::~AdjointVecData() {
  delete ids;
}


PetscErrorCode AdjointVecData::createAdjoint() {
  PetscCall(createFunc(&adjoint));
  PetscCall(initFunc(adjoint));
  PetscCall(VecSetSizes(adjoint, ids->getArraySize(), global_size));

  return PETSC_SUCCESS;
}

PetscErrorCode AdjointVecData::freeAdjoint() {
  PetscCall(VecDestroy(&adjoint));

  return PETSC_SUCCESS;
}

PetscErrorCode AdjointVecData::getAdjoint( VectorInterface* vi, PetscInt dim) {

  PetscCall(getAdjointNoReset(vi, dim));


  Identifier* data = ids->getArray();
  int         data_size = ids->getArraySize();
  // Reset afterwards since ids can contain duplicates.
  for(int i = 0; i < data_size; i += 1) {
    vi->resetAdjoint(data[i], dim);
  }
  ids->restoreArray(data);

  return PETSC_SUCCESS;
}

PetscErrorCode AdjointVecData::getAdjointNoReset(VectorInterface* vi, PetscInt dim) {
  auto func = [&](PetscInt AP_U(row), Real& value, Identifier id) {
    value = vi->getAdjoint(id, dim);
  };
  PetscCall(VecIterateAllEntries(func, adjoint, ids));

  return PETSC_SUCCESS;
}

PetscErrorCode AdjointVecData::updateAdjoint(VectorInterface* vi, PetscInt dim) {
  Tape& tape = Number::getTape();

  auto func = [&](PetscInt AP_U(row), Real& value, Identifier id) {
    if( tape.isIdentifierActive(id)) {
        vi->updateAdjoint(id, dim, value);
    }
  };
  PetscCall(VecIterateAllEntries(func, adjoint, ids));

  return PETSC_SUCCESS;
}

Vec AdjointVecData::getVec() {
  return adjoint;
}

AP_NAMESPACE_END
