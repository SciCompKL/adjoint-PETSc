#pragma once

#include "base.hpp"
#include "codi_def.hpp"

AP_NAMESPACE_START

struct WrapperArray {

  private:
    Real*       values_;
    Identifier* identifiers_;

  public:

    WrapperArray() = default;
    WrapperArray(Real* values, Identifier* identifiers) :
        values_(values),
        identifiers_(identifiers)
    {}


    Wrapper operator[](size_t i) {
      return Wrapper(values_[i], identifiers_[i]);
    }

    ConstWrapper const operator[](size_t i) const {
      return ConstWrapper(values_[i], identifiers_[i]);
    }

    Real* getValues() {
      return values_;
    }
};

AP_NAMESPACE_END
