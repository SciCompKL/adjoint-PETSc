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
