#pragma once

#include <adjoint_petsc/vec.h>

AP_NAMESPACE_START

namespace iterator_implementation {
  template<typename T, typename = void>
  struct VecValueAccess {

    T& value(PetscInt offset);
  };

  template<>
  struct VecValueAccess<Vec> {
    Vec vec;
    Real* values;

    VecValueAccess(Vec vec) : vec(vec), values(nullptr) {
      PetscCallVoid(VecGetArray(vec, &values));
    }

    ~VecValueAccess() {
      PetscCallVoid(VecRestoreArray(vec, &values));
    }

    Real& value(PetscInt offset) { return values[offset]; }
  };

  template<>
  struct VecValueAccess<ADVec> {
    ADVec vec;
    WrapperArray values;

    std::array<char, sizeof(Wrapper)> wrapper;

    VecValueAccess(ADVec vec) : vec(vec), values() {
      PetscCallVoid(VecGetArray(vec, &values));
    }

    ~VecValueAccess() {
      PetscCallVoid(VecRestoreArray(vec, &values));
    }

    Wrapper& value(PetscInt offset) {
      // TODO: Improve wrapper.
      Wrapper temp = values[offset];
      new (wrapper.data()) Wrapper(temp.value(), temp.getIdentifier());

      return *reinterpret_cast<Wrapper*>(wrapper.data());
    }
  };

  template<>
  struct VecValueAccess<Identifier*> {
    Identifier* ids;

    VecValueAccess(Identifier* ids) : ids(ids) {}

    Identifier& value(PetscInt offset) { return ids[offset]; }
  };

  template<typename Func, typename ... Values>
  PetscErrorCode iterateVecAll(Func&& func, Vec vec, VecValueAccess<Values>&& ... values) {

    PetscInt low;
    PetscInt high;
    PetscInt range;

    PetscCall(VecGetOwnershipRange(vec, &low, &high));
    range = high - low;

    for(PetscInt i = 0; i < range; i += 1) {
      func(i + low, values.value(i)...);
    }
    return PETSC_SUCCESS;
  }

  template<typename Func, typename ... Values>
  PetscErrorCode iterateVecIndexSet(Func&& func, PetscInt n, PetscInt const* ix, Vec vec, VecValueAccess<Values>&& ... values) {

    PetscInt low;
    PetscCall(VecGetOwnershipRange(vec, &low, nullptr));

    for(PetscInt i = 0; i < n; i += 1) {
      func(i, ix[i] + low, values.value(ix[i])...);
    }
    return PETSC_SUCCESS;
  }

  inline Vec getUnderlyingVec(Vec   vec) { return vec; }
  inline Vec getUnderlyingVec(ADVec vec) { return vec->vec; }
}

template<typename Func, typename First, typename ... Other>
PetscErrorCode VecIterateAllEntries(Func&& func, First&& vec, Other&& ... other) {
  // TODO: Check vector type and select iterator

  return iterator_implementation::iterateVecAll(
      func,
      iterator_implementation::getUnderlyingVec(std::forward<First>(vec)),
      iterator_implementation::VecValueAccess<std::remove_cvref_t<First>>(std::forward<First>(vec)),
      iterator_implementation::VecValueAccess<std::remove_cvref_t<Other>>(std::forward<Other>(other))...
    );
}

template<typename Func, typename First, typename ... Other>
PetscErrorCode VecIterateIndexSet(Func&& func, PetscInt n, PetscInt const* ix, First&& vec, Other&& ... other) {
  // TODO: Check vector type and select iterator

  return iterator_implementation::iterateVecIndexSet(
      func,
      n,
      ix,
      iterator_implementation::getUnderlyingVec(std::forward<First>(vec)),
      iterator_implementation::VecValueAccess<std::remove_cvref_t<First>>(std::forward<First>(vec)),
      iterator_implementation::VecValueAccess<std::remove_cvref_t<Other>>(std::forward<Other>(other))...
      );
}

AP_NAMESPACE_END
