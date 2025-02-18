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

    auto dyadic_init = [&] (PetscInt row, PetscInt col, Real& value) {
      if( col < low || high <= col) {
        col_list.insert(col);
      }
    };
    PetscCall(PetscObjectIterateAllEntries(dyadic_init, A));

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
