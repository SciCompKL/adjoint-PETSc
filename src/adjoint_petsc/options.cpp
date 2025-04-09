#include <iostream>

#include <adjoint_petsc/options.h>

AP_NAMESPACE_START
struct ADPetscOptions {
  bool          debug_output_primal      = false;
  bool          debug_output_reverse     = true;
  bool          debug_output_identifiers = true;
  int           debug_output_precission  = 12;
  std::ostream* debug_output_stream      = &std::cerr;
};

ADPetscOptions ad_gobal_options = {};

bool          ADPetscOptionsGetDebugOutputPrimal() { return ad_gobal_options.debug_output_primal; }
bool          ADPetscOptionsGetDebugOutputReverse() { return ad_gobal_options.debug_output_reverse; }
bool          ADPetscOptionsGetDebugOutputIdentifiers() { return ad_gobal_options.debug_output_identifiers; }
int           ADPetscOptionsGetDebugOutputPrecission() { return ad_gobal_options.debug_output_precission; }
std::ostream& ADPetscOptionsGetDebugOutputStream() { return *ad_gobal_options.debug_output_stream; }

void ADPetscOptionsSetDebugOutputPrimal(bool value) { ad_gobal_options.debug_output_primal = value; }
void ADPetscOptionsSetDebugOutputReverse(bool value) { ad_gobal_options.debug_output_reverse = value; }
void ADPetscOptionsSetDebugOutputIdentifiers(bool value) { ad_gobal_options.debug_output_identifiers = value; }
void ADPetscOptionsSetDebugOutputPrecission(int value) { ad_gobal_options.debug_output_precission = value; }
void ADPetscOptionsSetDebugOutputStream(std::ostream& value) { ad_gobal_options.debug_output_stream = &value; }

AP_NAMESPACE_END
