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
