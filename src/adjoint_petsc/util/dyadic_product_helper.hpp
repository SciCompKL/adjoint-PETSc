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

#pragma once

#include <petscsys.h>
#include <petscvec.h>
#include <petscmat.h>

#include <adjoint_petsc/util/base.hpp>
#include <adjoint_petsc/util/codi_def.hpp>
#include <adjoint_petsc/util/petsc_missing.h>

#include "mat_iterator_util.hpp"

AP_NAMESPACE_START

struct DyadicProductHelper {

  PetscInt low;
  PetscInt high;

  std::set<PetscInt> col_list;
  std::map<PetscInt, PetscInt> colmap;
  std::vector<PetscInt> col_vec;

  std::vector<Real> remote_values;

  PetscErrorCode init(Mat A, Vec vec) {
    PetscCall(VecGetOwnershipRange(vec, &low, &high));

    auto dyadic_init = [&] (PetscInt AP_U(row), PetscInt col, Real& AP_U(value)) {
      if( col < low || high <= col) {
        col_list.insert(col);
      }
    };
    PetscCall(MatIterateAllEntries(dyadic_init, A));

    col_vec.reserve(col_list.size());

    col_vec.insert(col_vec.begin(), col_list.begin(), col_list.end());

    PetscInt pos = 0;
    for(PetscInt col : col_vec) {
      colmap[col] = pos;
      pos += 1;
    }

    remote_values.resize(col_vec.size());

    return PETSC_SUCCESS;
  }

  PetscErrorCode communicateValues(Vec vec) {
    return VecGetValuesNonLocal(vec, col_vec.size(), col_vec.data(), remote_values.data());
  }

  PetscErrorCode getValue(Vec vec, PetscInt col, Real* val) {
    if( low <= col && col < high) {
      PetscCall(VecGetValues(vec, 1, &col, val));
    } else {
      *val = remote_values[colmap[col]];
    }

    return PETSC_SUCCESS;
  }
};

AP_NAMESPACE_END
