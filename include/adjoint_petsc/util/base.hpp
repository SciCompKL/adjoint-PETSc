#pragma once

#define AP_NAMESPACE adjoint_petsc
#define AP_NAMESPACE_START namespace AP_NAMESPACE {
#define AP_NAMESPACE_END }

AP_NAMESPACE_START

template<typename... Args>
inline void AP_UNUSED(Args const&...) {}

#define AP_U(arg) /* arg */

AP_NAMESPACE_END
