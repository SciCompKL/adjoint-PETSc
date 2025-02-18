#pragma once

#include "../mat.h"

AP_NAMESPACE_START


namespace iterator_implementation {

  template<typename Func>
  PetscErrorCode MatSeqAIJAccessValue(Mat mat, PetscInt const* colmap, Identifier* ad_data, PetscInt row, PetscInt col, Func&& func) {
    PetscInt const* row_offset;
    PetscInt const* column_ids;
    Real* values;
    PetscMemType mem;

    PetscCall(MatSeqAIJGetCSRAndMemType(mat, &row_offset, &column_ids, &values, &mem));

    int cur_local_row = row;
    int col_range = row_offset[cur_local_row + 1] - row_offset[cur_local_row];
    int col_offset = row_offset[cur_local_row];

    for(int cur_local_col = 0; cur_local_col < col_range; cur_local_col += 1) {
      int cur_col = column_ids[cur_local_col + col_offset];
      if(nullptr != colmap) {
        cur_col = colmap[cur_col];
      }

      if(cur_col == col) {
        Wrapper element = createRefType(values[cur_local_col + col_offset], ad_data[cur_local_col + col_offset]);
        func(element);

        return PETSC_SUCCESS;
      }
    }

    return PETSC_ERR_USER_INPUT;
  }

  template<typename Func>
  PetscErrorCode MatAIJAccessValue(Mat mat, Identifier* ad_data_d, Identifier* ad_data_o, PetscInt row, PetscInt col, Func&& func) {
    Mat matd;
    Mat mato;
    PetscInt const* colmap;
    PetscCall(MatMPIAIJGetSeqAIJ(mat, &matd, &mato, &colmap));

    PetscInt col_low;
    PetscInt col_high;
    PetscCall(MatGetOwnershipRangeColumn(mat, &col_low, &col_high));
    PetscInt row_low;
    PetscInt row_high;
    PetscCall(MatGetOwnershipRange(mat, &row_low, &row_high));

    if(col_low <= col && col < col_high) {
      PetscCall(MatSeqAIJAccessValue(matd, nullptr, ad_data_d, row - row_low, col - col_low, std::forward<Func>(func)));
    } else {
      PetscCall(MatSeqAIJAccessValue(mato, colmap, ad_data_o, row - row_low, col, std::forward<Func>(func)));
    }

    return PETSC_SUCCESS;
  }

  template<typename T, typename = void>
  struct MatSeqAIJValueAccess {

    T& value(PetscInt offset);
  };

  template<>
  struct MatSeqAIJValueAccess<Mat> {
    Real* values;

    MatSeqAIJValueAccess(Mat mat) : values(nullptr) {
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat, nullptr, nullptr, &values, nullptr));
    }

    Real& value(PetscInt offset) { return values[offset]; }
  };

  template<>
  struct MatSeqAIJValueAccess<ADMat> {
    MatSeqAIJValueAccess<Mat> mat_v;
    ADMat                     mat;

    std::array<char, sizeof(Wrapper)> wrapper;

    MatSeqAIJValueAccess(ADMat mat) : mat_v(mat->mat), mat(mat) {}

    Wrapper& value(PetscInt offset) {
      new (wrapper.data()) Wrapper(mat_v.value(offset), mat->ad_data[offset]);

      return *reinterpret_cast<Wrapper*>(wrapper.data());
    }
  };

  template<typename Func, typename ... Values>
  PetscErrorCode iterateMatSeqAIJI(Func&& func, Mat mat, PetscInt const* colmap, MatSeqAIJValueAccess<Values>&& ... values) {

    PetscInt const* row_offset;
    PetscInt const* column_ids;
    PetscMemType mem;
    PetscInt low;
    PetscInt high;
    PetscInt row_range;

    PetscCall(MatGetOwnershipRange(mat, &low, &high));
    PetscCall(MatSeqAIJGetCSRAndMemType(mat, &row_offset, &column_ids, nullptr, &mem));

    row_range = high - low;

    for(int cur_local_row = 0; cur_local_row < row_range; cur_local_row += 1) {
      int col_range = row_offset[cur_local_row + 1] - row_offset[cur_local_row];
      int col_offset = row_offset[cur_local_row];

      int row = cur_local_row + low;
      for(int cur_local_col = 0; cur_local_col < col_range; cur_local_col += 1) {
        int col = column_ids[cur_local_col + col_offset];
        if(nullptr != colmap) {
          col = colmap[col];
        } else {
          col += low;
        }
        func(row, col, values.value(cur_local_col + col_offset)...);
      }
    }

    return PETSC_SUCCESS;
  }

  template<typename T, typename = void>
  struct MatAIJValueAccess {

    MatSeqAIJValueAccess<T> accessMatD();
    MatSeqAIJValueAccess<T> accessMatO();
  };

  template<>
  struct MatAIJValueAccess<Mat> {
    Mat mat;

    Mat matd;
    Mat mato;

    MatAIJValueAccess() = default;

    MatAIJValueAccess(Mat mat_) : MatAIJValueAccess() {
      mat = mat_;
      PetscCallVoid(MatMPIAIJGetSeqAIJ(mat, &matd, &mato, nullptr));
    }

    auto accessMatD() { return MatSeqAIJValueAccess<Mat>(matd); };
    auto accessMatO() { return MatSeqAIJValueAccess<Mat>(mato); };
  };

  template<>
  struct MatAIJValueAccess<ADMat> {
    ADMat mat;

    ADMatImpl matd;
    ADMatImpl mato;

    MatAIJValueAccess() = default;

    MatAIJValueAccess(ADMat mat_) : MatAIJValueAccess() {
      mat = mat_;

      PetscCallVoid(MatMPIAIJGetSeqAIJ(mat->mat, &matd.mat, &mato.mat, nullptr));

      // TODO: Very hacky. Improve when handling multiple data formats in ADMat.
      matd.ad_data = mat->ad_data;
      matd.ad_size = mat->ad_size_diag;

      mato.ad_data = &mat->ad_data[mat->ad_size_diag];
      mato.ad_size = mat->ad_size - mat->ad_size_diag;
    }

    auto accessMatD() { return MatSeqAIJValueAccess<ADMat>(&matd); };
    auto accessMatO() { return MatSeqAIJValueAccess<ADMat>(&mato); };
  };


  template<typename Func, typename ... Values>
  PetscErrorCode iterateMatAIJ(Func&& func, Mat mat, MatAIJValueAccess<Values>&& ... values) {

    Mat matd;
    Mat mato;
    PetscInt const* colmap;
    PetscCall(MatMPIAIJGetSeqAIJ(mat, &matd, &mato, &colmap));

    PetscInt col_low;
    PetscInt col_high;

    PetscInt row_low;
    PetscInt row_high;
    PetscCall(MatGetOwnershipRange(mat, &row_low, &row_high));
    PetscCall(MatGetOwnershipRangeColumn(mat, &col_low, &col_high));

    auto func_shift = [&] (PetscInt row, PetscInt col, auto&& ... values) {
      func(row + row_low, col + col_low, std::forward<decltype(values)>(values)...);
    };

    auto func_shift_row = [&] (PetscInt row, PetscInt col, auto&& ... values) {
      func(row + row_low, col, std::forward<decltype(values)>(values)...);
    };

    PetscCall(iterateMatSeqAIJI(func_shift,     matd, nullptr, values.accessMatD()...));
    PetscCall(iterateMatSeqAIJI(func_shift_row, mato, colmap, values.accessMatO()...));

    return PETSC_SUCCESS;
  }

  inline Mat getUnderlyingMat(Mat   mat) { return mat; }
  inline Mat getUnderlyingMat(ADMat mat) { return mat->mat; }
}

template<typename Func, typename First, typename ... Other>
PetscErrorCode PetscObjectIterateAllEntries(Func&& func, First&& mat, Other&& ... other) {
  // TODO: Check matrix type and select iterator

  return iterator_implementation::iterateMatAIJ(
      func,
      iterator_implementation::getUnderlyingMat(std::forward<First>(mat)),
      iterator_implementation::MatAIJValueAccess<std::remove_cvref_t<First>>(std::forward<First>(mat)),
      iterator_implementation::MatAIJValueAccess<std::remove_cvref_t<Other>>(std::forward<Other>(other))...
    );
}

template<typename Func>
PetscErrorCode ADMatAccessValue(ADMat mat, PetscInt row, PetscInt col, Func&& func) {
  // TODO: Check matrix type and select iterator

  return iterator_implementation::MatAIJAccessValue(mat->mat, mat->ad_data, &mat->ad_data[mat->ad_size_diag], row, col, std::forward<Func>(func));
}

AP_NAMESPACE_END
