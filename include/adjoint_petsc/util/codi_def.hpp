#pragma once

#include <codi.hpp>

#include "base.hpp"

AP_NAMESPACE_START

using Number       = codi::RealReverse;
using Real         = typename Number::Real;
using Identifier   = typename Number::Identifier;
using Tape         = typename Number::Tape;
using Wrapper      = typename codi::ActiveTypeWrapper<Number>;
using ConstWrapper = typename codi::ImmutableActiveType<Number>;

using VectorInterface = typename codi::VectorAccessInterface<Real, Identifier>;

inline Wrapper createRefType(Real& value, Identifier& identifier) {
  return Wrapper(value, identifier);
}


AP_NAMESPACE_END
