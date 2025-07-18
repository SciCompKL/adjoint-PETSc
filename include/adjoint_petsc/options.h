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

#include "util/base.hpp"

AP_NAMESPACE_START

bool          ADPetscOptionsGetDebugOutputPrimal();
bool          ADPetscOptionsGetDebugOutputReverse();
bool          ADPetscOptionsGetDebugOutputIdentifiers();
int           ADPetscOptionsGetDebugOutputPrecission();
std::ostream& ADPetscOptionsGetDebugOutputStream();

void ADPetscOptionsSetDebugOutputPrimal(bool value);
void ADPetscOptionsSetDebugOutputReverse(bool value);
void ADPetscOptionsSetDebugOutputIdentifiers(bool value);
void ADPetscOptionsSetDebugOutputPrecission(int value);
void ADPetscOptionsSetDebugOutputStream(std::ostream& value);

AP_NAMESPACE_END
