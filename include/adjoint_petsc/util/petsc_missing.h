#pragma once

#include "base.hpp"

#include <petscvec.h>
#include <petscmat.h>

AP_NAMESPACE_START

PetscErrorCode VecGetValuesNonLocal(Vec vec, PetscInt ni, PetscInt ix[], PetscScalar y[]);

PetscErrorCode MatGetColumnSumAbs(Mat mat, PetscScalar y[]);

AP_NAMESPACE_END
