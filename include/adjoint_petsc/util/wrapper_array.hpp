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

    Identifier* getIdentifiers() {
      return identifiers_;
    }
};

AP_NAMESPACE_END
