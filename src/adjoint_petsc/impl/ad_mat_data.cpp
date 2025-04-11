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


ADMatMPIAIJData::ADMatMPIAIJData(PetscInt diag_size, PetscInt off_diag_size) :
    ADMatData(TYPE),
    index_d(diag_size),
    index_o(off_diag_size) {}

ADMatMPIAIJData* ADMatMPIAIJData::clone() {
  return new ADMatMPIAIJData(*this);
}

ADMatMPIAIJData* ADMatMPIAIJData::cast(ADMatData* d) {
  d->checkType(TYPE);

  return dynamic_cast<ADMatMPIAIJData*>(d);
}

ADMatType ADMatDataPTypeToEnum(MatType ptype) {
  if(0 == strcmp(ptype, MATAIJ)) {
    int size;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if(1 == size) {
      return ADMatType::ADMatSeqAIJ;
    }
    else {
      return ADMatType::ADMatMPIAIJ;
    }
  }
  else if(0 == strcmp(ptype, MATMPIAIJ)) {
    return ADMatType::ADMatMPIAIJ;
  }
  else if(0 == strcmp(ptype, MATSEQAIJ)) {
    return ADMatType::ADMatSeqAIJ;
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
