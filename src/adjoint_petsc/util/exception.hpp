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

#include <adjoint_petsc/util/base.hpp>
#include <adjoint_petsc/util/codi_def.hpp>
#include <adjoint_petsc/vec.h>

AP_NAMESPACE_START

inline std::string v_toString(const char* format, va_list list) {
    const int bufferSize = 200;
    char buffer[bufferSize];

    // copy the list if we need to iterate through the variables again
    va_list listCpy;
    va_copy(listCpy, list);

    int outSize = vsnprintf(buffer, bufferSize, format, list);

    std::string result;
    if(outSize + 1 > bufferSize) {
        char* newBuffer = new char[outSize + 1];

        outSize = vsnprintf(newBuffer, outSize + 1, format, listCpy);

        result = newBuffer;

        delete [] newBuffer;
    } else {
        result = buffer;
    }

    // cleanup the copied list
    va_end (listCpy);

    return result;
}

inline void throwException(char const function[], char const file[],
                                    int const line, char const* message,
                                    ...) {

  va_list list;
  va_start(list, message);
  std::string result = v_toString(message, list);
  va_end(list);

  fprintf(stderr, "Error in function %s (%s:%d)\nThe message is: %s\n", function, file, line, result.c_str());

  throw std::runtime_error(result);
}

#define AP_EXCEPTION(...)  :: AP_NAMESPACE ::throwException(__func__, __FILE__, __LINE__, __VA_ARGS__)

AP_NAMESPACE_END
