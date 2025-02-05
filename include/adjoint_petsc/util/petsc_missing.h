#pragma once

#include "base.hpp"

#include <petscvec.h>

AP_NAMESPACE_START

PetscErrorCode VecGetValuesNonLocal(Vec vec, PetscInt ni, PetscInt ix[], PetscScalar y[]);

AP_NAMESPACE_END
