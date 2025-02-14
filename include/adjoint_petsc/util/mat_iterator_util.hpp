#pragma once

#include "../mat.h"

AP_NAMESPACE_START


namespace iterator_implementation {

  // TODO: Generalize for different sparsity pattern and with respect to other ierator functions. (Make them share code.)
  template<typename Func>
  PetscErrorCode MatSeqAIJIterateAllEntries(Mat mat, PetscInt const* colmap, Func&& func) {
    PetscInt const* row_offset;
    PetscInt const* column_ids;
    Real* values;
    PetscMemType mem;
    PetscInt low;
    PetscInt high;
    PetscInt row_range;

    PetscCall(MatGetOwnershipRange(mat, &low, &high));
    PetscCall(MatSeqAIJGetCSRAndMemType(mat, &row_offset, &column_ids, &values, &mem));

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
        func(row, col, values[cur_local_col + col_offset]);
      }
    }

    return PETSC_SUCCESS;
  }

  // TODO: Generalize for different sparsity pattern and with respect to other ierator functions. (Make them share code.)
  template<typename Func>
  PetscErrorCode MatAIJIterateAllEntries(Mat mat, Func&& func) {
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

    auto func_shift = [&] (PetscInt row, PetscInt col, PetscReal& value) {
      func(row + row_low, col + col_low, value);
    };

    auto func_shift_row = [&] (PetscInt row, PetscInt col, PetscReal& value) {
      func(row + row_low, col, value);
    };

    PetscCall(MatSeqAIJIterateAllEntries(matd, nullptr, func_shift));
    PetscCall(MatSeqAIJIterateAllEntries(mato, colmap, func_shift_row));

    return PETSC_SUCCESS;
  }

  template<typename Func>
  PetscErrorCode MatSeqAIJIterateAllEntries(Mat mat, PetscInt const* colmap, Identifier* ad_data, Func&& func) {
    PetscInt const* row_offset;
    PetscInt const* column_ids;
    Real* values;
    PetscMemType mem;
    PetscInt low;
    PetscInt high;
    PetscInt row_range;

    PetscCall(MatGetOwnershipRange(mat, &low, &high));
    PetscCall(MatSeqAIJGetCSRAndMemType(mat, &row_offset, &column_ids, &values, &mem));

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
        Wrapper element = createRefType(values[cur_local_col + col_offset], ad_data[cur_local_col + col_offset]);
        func(row, col, element);
      }
    }

    return PETSC_SUCCESS;
  }

  template<typename Func>
  PetscErrorCode MatAIJIterateAllEntries(Mat mat, Identifier* ad_data_d, Identifier* ad_data_o, Func&& func) {
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

    auto func_shift = [&] (PetscInt row, PetscInt col, Wrapper& value) {
      func(row + row_low, col + col_low, value);
    };

    auto func_shift_row = [&] (PetscInt row, PetscInt col, Wrapper& value) {
      func(row + row_low, col, value);
    };

    PetscCall(MatSeqAIJIterateAllEntries(matd, nullptr, ad_data_d, func_shift));
    PetscCall(MatSeqAIJIterateAllEntries(mato, colmap, ad_data_o, func_shift_row));

    return PETSC_SUCCESS;
  }

  // TODO: Generalize for different sparsity pattern and with respect to other ierator functions. (Make them share code.)
  template<typename Func>
  PetscErrorCode MatSeqAIJIterateAllEntries(Mat mat_a, Mat mat_b, PetscInt const* colmap, Identifier* ad_data_a, Identifier* ad_data_b, Func&& func) {
    PetscInt const* row_offset;
    PetscInt const* column_ids;
    Real* values_a;
    Real* values_b;
    PetscMemType mem;
    PetscInt low;
    PetscInt high;
    PetscInt row_range;

    PetscCall(MatGetOwnershipRange(mat_a, &low, &high));
    PetscCall(MatSeqAIJGetCSRAndMemType(mat_a, &row_offset, &column_ids, &values_a, &mem));
    PetscCall(MatSeqAIJGetCSRAndMemType(mat_b, NULL, NULL, &values_b, NULL));

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
        Wrapper element_a = createRefType(values_a[cur_local_col + col_offset], ad_data_a[cur_local_col + col_offset]);
        Wrapper element_b = createRefType(values_b[cur_local_col + col_offset], ad_data_b[cur_local_col + col_offset]);
        func(row, col, element_a, element_b);
      }
    }

    return PETSC_SUCCESS;
  }

  // TODO: Generalize for different sparsity pattern and with respect to other ierator functions. (Make them share code.)
  template<typename Func>
  PetscErrorCode MatAIJIterateAllEntries(Mat mat_a, Identifier* ad_data_a_d, Identifier* ad_data_a_o, Mat mat_b, Identifier* ad_data_b_d, Identifier* ad_data_b_o, Func&& func) {
    Mat matd_a;
    Mat mato_a;
    PetscInt const* colmap;
    PetscCall(MatMPIAIJGetSeqAIJ(mat_a, &matd_a, &mato_a, &colmap));

    Mat matd_b;
    Mat mato_b;
    PetscCall(MatMPIAIJGetSeqAIJ(mat_b, &matd_b, &mato_b, &colmap));

    PetscInt col_low;
    PetscInt col_high;
    PetscCall(MatGetOwnershipRangeColumn(mat_a, &col_low, &col_high));
    PetscInt row_low;
    PetscInt row_high;
    PetscCall(MatGetOwnershipRange(mat_a, &row_low, &row_high));

    auto func_shift = [&] (PetscInt row, PetscInt col, Wrapper& value_a, Wrapper& value_b) {
      func(row + row_low, col + col_low, value_a, value_b);
    };

    auto func_shift_row = [&] (PetscInt row, PetscInt col, Wrapper& value_a, Wrapper& value_b) {
      func(row + row_low, col, value_a, value_b);
    };

    PetscCall(MatSeqAIJIterateAllEntries(matd_a, matd_b, nullptr, ad_data_a_d, ad_data_b_d, func_shift));
    PetscCall(MatSeqAIJIterateAllEntries(mato_a, mato_b, colmap, ad_data_a_o, ad_data_b_o, func_shift_row));

    return PETSC_SUCCESS;
  }

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
}

  template<typename Func>
  PetscErrorCode PetscObjectIterateAllEntries(Mat mat, Func&& func) {
    // TODO: Check matrix type and select iterator

    return iterator_implementation::MatAIJIterateAllEntries(mat, std::forward<Func>(func));
  }

template<typename Func>
PetscErrorCode ADObjIterateAllEntries(ADMat mat, Func&& func) {
  // TODO: Check matrix type and select iterator

  return iterator_implementation::MatAIJIterateAllEntries(mat->mat, mat->ad_data, &mat->ad_data[mat->ad_size_diag], std::forward<Func>(func));
}

template<typename Func>
PetscErrorCode ADObjIterateAllEntries(ADMat mat_a, ADMat mat_b, Func&& func) {
  // TODO: Check matrix type and select iterator

  return iterator_implementation::MatAIJIterateAllEntries(mat_a->mat, mat_a->ad_data, &mat_a->ad_data[mat_a->ad_size_diag],
                                                          mat_b->mat, mat_b->ad_data, &mat_b->ad_data[mat_b->ad_size_diag],
                                                          std::forward<Func>(func));
}

template<typename Func>
PetscErrorCode ADMatAccessValue(ADMat mat, PetscInt row, PetscInt col, Func&& func) {
  // TODO: Check matrix type and select iterator

  return iterator_implementation::MatAIJAccessValue(mat->mat, mat->ad_data, &mat->ad_data[mat->ad_size_diag], row, col, std::forward<Func>(func));
}

AP_NAMESPACE_END
