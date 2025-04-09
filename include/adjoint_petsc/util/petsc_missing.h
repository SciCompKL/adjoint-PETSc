#pragma once

#include "base.hpp"

#include <petscvec.h>
#include <petscmat.h>

AP_NAMESPACE_START

/*-------------------------------------------------------------------------------------------------
 * Vec functions
 */

PetscErrorCode VecGetValuesNonLocal(Vec vec, PetscInt ni, PetscInt ix[], PetscScalar y[]);

/*-------------------------------------------------------------------------------------------------
 * Mat functions
 */

PetscErrorCode MatGetColumnSumAbs(Mat mat, PetscScalar y[]);

AP_NAMESPACE_END
