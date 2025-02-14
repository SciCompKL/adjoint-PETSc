#include "../../include/adjoint_petsc/mat.h"

#include "../../include/adjoint_petsc/util/mat_iterator_util.hpp"
#include "../../include/adjoint_petsc/util/addata_helper.hpp"
#include "../../include/adjoint_petsc/util/petsc_missing.h"

AP_NAMESPACE_START

struct LhsInData {
  PetscInt row;
  PetscInt col;
};

bool operator <(LhsInData const& a, LhsInData const& b) {
  return a.row < b.row || (a.row == b.row && a.col < b.col);
}

bool operator ==(LhsInData const& a, LhsInData const& b) {
  return a.row == b.row && a.col == b.col;
}

struct CommunicationData {
  MPI_Comm comm;
  std::vector<int> request_number_for_rank_send;
  std::vector<int> request_number_for_rank_recv;

  int total_request_send;
  int total_request_recv;

  std::vector<int> send_displs;
  std::vector<int> recv_displs;

  CommunicationData() : comm(MPI_COMM_SELF), request_number_for_rank_send(0), request_number_for_rank_recv(0),
      total_request_send(0), total_request_recv(0), send_displs(0), recv_displs(0) {}

  void initialize(Mat mat, std::vector<LhsInData> const& lhs_in_positions) {
    PetscInt const* ownership_ranges;
    PetscObjectGetComm((PetscObject)mat, &comm);
    PetscCallVoid(MatGetOwnershipRanges(mat, &ownership_ranges));

    // Step 1: Get number of requested values per rank
    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    // The vector with the lhs_in positions is sorted by rows.
    request_number_for_rank_send.resize(mpi_size);
    int cur_rank = 0;
    for(LhsInData const& cur : lhs_in_positions) {
      // Adjust rank
      while(cur.row >= ownership_ranges[cur_rank + 1]) {
        cur_rank += 1;
      }

      // Add entries
      request_number_for_rank_send[cur_rank] += 1;
    }

    request_number_for_rank_recv.resize(mpi_size);
    MPI_Alltoall(request_number_for_rank_send.data(), 1, MPI_INTEGER, request_number_for_rank_recv.data(), 1, MPI_INTEGER, comm);

    // Step 2: Communicate requesteed values to ranks
    // step 2.1: Compute the total size and allocate the send/recv displs
    total_request_send = 0;
    total_request_recv = 0;
    send_displs.resize(mpi_size + 1); // One extra for total displacement.
    recv_displs.resize(mpi_size + 1); // One extra for total displacement.
    for(int i = 0; i < mpi_size; i += 1) {
      send_displs[i] = total_request_send;
      recv_displs[i] = total_request_recv;
      total_request_send += request_number_for_rank_send[i];
      total_request_recv += request_number_for_rank_recv[i];
    }
    send_displs[mpi_size] = total_request_send;
    recv_displs[mpi_size] = total_request_recv;
  }

  void commRowCol(LhsInData* send, LhsInData* recv) {
    MPI_Datatype position_type;
    MPI_Type_contiguous(2, MPI_INTEGER, &position_type);
    MPI_Type_commit(&position_type);

    MPI_Alltoallv(send, request_number_for_rank_send.data(), send_displs.data(), position_type,
                  recv, request_number_for_rank_recv.data(), recv_displs.data(), position_type,
                  comm);

    MPI_Type_free(&position_type);
  }

  void commReverse(Real* adjoints_recv, Real* adjoints_send, MPI_Datatype adjoint_type) {
    MPI_Alltoallv(adjoints_recv, request_number_for_rank_recv.data(), recv_displs.data(), adjoint_type,
                  adjoints_send, request_number_for_rank_send.data(), send_displs.data(), adjoint_type,
                  comm);
  }
};

struct ADData_MatSetValues {
  struct RhsData {
    PetscInt lhs_pos;
    Identifier rhs_identifier;
  };

  std::vector<Identifier> lhs_identifiers;
  std::vector<int> lhs_out_positions;

  std::vector<LhsInData> lhs_in_positions;
  std::vector<RhsData> rhs_data;

  std::vector<int> step_boundaries;

  InsertMode mode;

  std::vector<Real> adjoint_data;
  int adjoint_step;

  CommunicationData comm_data;

  ADData_MatSetValues(InsertMode m) : lhs_identifiers(0), lhs_out_positions(0), lhs_in_positions(0),
      rhs_data(0), step_boundaries(0), mode(m), adjoint_data(0), adjoint_step(), comm_data() {
  }

  void addEntries(PetscInt m, PetscInt const* idxm, PetscInt n, PetscInt const* idxn, Number const* values) {
    step_boundaries.push_back((int)rhs_data.size());
    for (int cur_row = 0; cur_row < m; cur_row += 1) {
      for (int cur_col = 0; cur_col < n; cur_col += 1) {
        addEntry(idxm[cur_row], idxn[cur_col], values[cur_row * n + cur_col]);
      }
    }

    if(Number::getTape().isActive()) {
      Number::getTape().pushExternalFunction(codi::ExternalFunction<Tape>::create(&step_reverse, this, &no_delete));
    }
  }

  void finalize(ADMat mat) {
    step_boundaries.push_back((int)rhs_data.size());

    finalizeData(); // Sort the lhs_in_positions for better access

    comm_data.initialize(mat->mat, lhs_in_positions);

    std::vector<LhsInData> row_col_data(comm_data.total_request_recv);
    comm_data.commRowCol(lhs_in_positions.data(), row_col_data.data());

    std::vector<LhsInData> row_col_data_unique(row_col_data);
    lhs_out_positions.resize(comm_data.total_request_recv);

    std::sort(row_col_data_unique.begin(), row_col_data_unique.end());
    auto unique_end = std::unique(row_col_data_unique.begin(), row_col_data_unique.end());
    row_col_data_unique.erase(unique_end, row_col_data_unique.end());

    int lhs_out_pos = 0;
    int cur_rank = 0;
    for(int i = 0; i < comm_data.total_request_recv; i += 1) {
      LhsInData const& cur = row_col_data[i];

      // Reset lhs_out_pos on rank shift
      while(i >= comm_data.recv_displs[cur_rank + 1]) {
        cur_rank += 1;
        lhs_out_pos = 0;
      }

      // Now search in the lhs_out values
      for(; lhs_out_pos < (int)row_col_data_unique.size(); lhs_out_pos += 1) {
        LhsInData const& cur_unique = row_col_data_unique[lhs_out_pos];
        if(cur.row == cur_unique.row && cur.col == cur_unique.col) {
          lhs_out_positions[i] = lhs_out_pos;
          break;
        }
      }
    }

    // Register all modified values
    lhs_identifiers.resize(row_col_data_unique.size());
    Tape& tape = Number::getTape();
    int pos = 0;
    for(LhsInData const& cur : row_col_data_unique) {
      auto func = [&](Wrapper& value) {

        if(tape.isActive()) {
          tape.registerExternalFunctionOutput(value);
          lhs_identifiers[pos] = value.getIdentifier();
        } else {
          tape.deactivateValue(value);
        }
      };
      PetscCallVoid(ADMatAccessValue(mat, cur.row, cur.col, func));

      pos += 1;
    }

    if(tape.isActive()) {
      Number::getTape().pushExternalFunction(codi::ExternalFunction<Tape>::create(&assemble_reverse, this, &ad_delete));
    } else {
      delete this;
    }
  }

  static void step_reverse(Tape* tape, void* d, VectorInterface* va) {
    ADData_MatSetValues* data = (ADData_MatSetValues*)d;
    int ad_vec_size = va->getVectorSize();

    data->adjoint_step -= 1;
    int from = data->step_boundaries[data->adjoint_step];
    int to = data->step_boundaries[data->adjoint_step + 1];

    for(int i = from; i < to; i += 1) {
      va->updateAdjointVec(data->rhs_data[i].rhs_identifier, &data->adjoint_data[data->rhs_data[i].lhs_pos * ad_vec_size]);
    }

    if(data->adjoint_step == 0) {
      data->adjoint_data.resize(0);
    }
  }

  static void assemble_reverse(Tape* tape, void* d, VectorInterface* va) {
    ADData_MatSetValues* data = (ADData_MatSetValues*)d;
    int ad_vec_size = va->getVectorSize();

    data->adjoint_step = (int)data->step_boundaries.size() - 1;

    data->communicateADData(va);
  }

  void communicateADData(VectorInterface* va) {
    int vec_size = va->getVectorSize();
    std::vector<Real> adjoints_recv(comm_data.total_request_recv * vec_size);
    for(int i = 0; i < comm_data.total_request_recv; i += 1) {
      Identifier identifier = lhs_identifiers[lhs_out_positions[i]];
      va->getAdjointVec(identifier, &adjoints_recv[i * vec_size]);
    }

    // Step 4: Clean adjoints
    for(Identifier const& cur : lhs_identifiers) {
      va->resetAdjointVec(cur);
    }

    // Step 5: Send requested adjoint values to ranks
    MPI_Datatype adjoint_type = MPI_DOUBLE;
    if( 1 != vec_size) {
      MPI_Type_contiguous(vec_size, MPI_DOUBLE, &adjoint_type);
      MPI_Type_commit(&adjoint_type);
    }

    adjoint_data.resize(comm_data.total_request_send * vec_size);
    comm_data.commReverse(adjoints_recv.data(), adjoint_data.data(), adjoint_type);

    if( 1 != vec_size) {
      MPI_Type_free(&adjoint_type);
    }
  }

  static void no_delete(Tape* tape, void* data) {
    (void)tape;
    (void)data;
  }

  static void ad_delete(Tape* tape, void* d) {
    (void)tape;
    ADData_MatSetValues* data = (ADData_MatSetValues*)d;

    delete data;
  }


  private:
  void addEntry(PetscInt row, PetscInt col, Number const& rhs_value) {
    PetscInt lhs_cache_pos = insertLhsPos(row, col);

    rhs_data.push_back({lhs_cache_pos, rhs_value.getIdentifier()});
  }

  PetscInt insertLhsPos(PetscInt row, PetscInt col) {
    // Search for the position.
    for(size_t i = 0; i < lhs_in_positions.size(); i += 1) {
      if(row == lhs_in_positions[i].row && col == lhs_in_positions[i].col) {
        return i;
      }
    }

    // Did not find the position add it.
    lhs_in_positions.push_back({row, col});

    return (PetscInt)lhs_in_positions.size() - 1;
  }

  void finalizeData() {
    // We sort an index array, which gives us the lookup to update the stored positions in the rhs entries.
    std::vector<PetscInt> lookup(lhs_in_positions.size());
    PetscInt i = 0;
    for(PetscInt& cur : lookup) { cur = i; i += 1; }

    // Access the actual data for comparison.
    auto compare = [&](PetscInt a_i, PetscInt b_i) -> bool {
      LhsInData const& a = lhs_in_positions[a_i];
      LhsInData const& b = lhs_in_positions[b_i];

      return a < b;

    };

    // Do the sort.
    std::sort(lookup.begin(), lookup.end(), compare);

    // Now 'sort' the lhs data
    std::vector<LhsInData> copy = lhs_in_positions;
    for(size_t i = 0; i < copy.size(); i += 1) {
      lhs_in_positions[i] = copy[lookup[i]];
    }

    // Build the reverse lookup table
    std::vector<PetscInt> lookup_reverse(lhs_in_positions.size());
    i = 0;
    for(PetscInt& cur : lookup) {
      lookup_reverse[lookup[i]] = i;
      i += 1;
    }

    // Update the lhs positions for the rhs data.
    for(RhsData& cur : rhs_data) {
      cur.lhs_pos = lookup_reverse[cur.lhs_pos];
    }
  }
};


PetscErrorCode MatAssemblyBegin(ADMat mat, MatAssemblyType type) {
  return MatAssemblyBegin(mat->mat, type);
}

PetscErrorCode MatAssemblyEnd(ADMat mat, MatAssemblyType type) {
  PetscCall(MatAssemblyEnd(mat->mat, type));

  ADMatCreateADData(mat);

  if(nullptr != mat->transaction_data) {
    ADData_MatSetValues* data = reinterpret_cast<ADData_MatSetValues*>(mat->transaction_data);

    data->finalize(mat);
  }

  return PETSC_SUCCESS;
}
// PetscErrorCode MatConvert                (ADMat mat, MatType newtype, MatReuse reuse, ADMat *M);
// PetscErrorCode MatCreate                 (MPI_Comm comm, ADMat* mat);
PetscErrorCode MatCreateAIJ(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], ADMat *A) {
  (*A) = new ADMatImpl();
  PetscCall(MatCreateAIJ(comm, m, n, M, N, d_nz, d_nnz, o_nz, o_nnz, &(*A)->mat));

  return PETSC_SUCCESS;
}
// PetscErrorCode MatCreateDense            (MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Number *data, ADMat *A);

PetscErrorCode MatDestroy(ADMat* mat) {

  if((*mat)->ad_size != 0) {
    Tape& tape = Number::getTape();
    ADObjIterateAllEntries(*mat, [&](PetscInt row, PetscInt col, Wrapper& el) {
      (void)row;
      (void)col;
      tape.deactivateValue(el);
    });

    delete [] (*mat)->ad_data;
  }

  PetscCall(MatDestroy(&(*mat)->mat));
  delete *mat;
  *mat = nullptr;

  return PETSC_SUCCESS;
}

PetscErrorCode MatDuplicate(ADMat mat, MatDuplicateOption op, ADMat* newv) {
  *newv = new ADMatImpl();

  PetscCall(MatDuplicate(mat->mat, op, &(*newv)->mat));

  ADMatCreateADData(*newv);

  if(op == MAT_COPY_VALUES) {
    auto func = [&] (PetscInt row, PetscInt col, Wrapper& a, Wrapper& b) {
      b = a;
    };
    PetscCall(ADObjIterateAllEntries(mat, *newv, func));
  }

  return PETSC_SUCCESS;

}

PetscErrorCode MatGetInfo(ADMat mat, MatInfoType flag, MatInfo *info) {
  return MatGetInfo(mat->mat, flag, info);
}

PetscErrorCode MatGetLocalSize(ADMat mat, PetscInt *m, PetscInt *n) {
  return MatGetLocalSize(mat->mat, m, n);
}

PetscErrorCode MatGetOwnershipRange(ADMat mat, PetscInt *m, PetscInt *n) {
  return MatGetOwnershipRange(mat->mat, m, n);
}

PetscErrorCode MatGetSize(ADMat mat, PetscInt *m, PetscInt *n) {
  return MatGetSize(mat->mat, m, n);
}

PetscErrorCode MatGetType                (ADMat mat, MatType *type) {
  return MatGetType(mat->mat, type);
}

PetscErrorCode MatGetValues(ADMat mat, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], Number v[]) {
  for(PetscInt row = 0; row < m; row += 1) {
    if(idxm[row] < 0) { continue; }

    for(PetscInt col = 0; col < n; col += 1) {
      if (idxn[col] < 0) { continue; }

      auto func = [&] (Wrapper& wrap) {
        v[row * n + col] = wrap;
      };
      PetscCall(ADMatAccessValue(mat, idxm[row], idxn[col], func));
    }
  }

  return PETSC_SUCCESS;
}

PetscErrorCode MatGetValue(ADMat mat, PetscInt row, PetscInt col, Number* v) {
  return MatGetValues(mat, 1, &row, 1, &col, v);
}

PetscErrorCode MatMPIAIJSetPreallocation(ADMat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[]) {
  return MatMPIAIJSetPreallocation(B->mat, d_nz, d_nnz, o_nz, o_nnz);
}

struct ADData_MatMult : public ReverseDataBase<ADData_MatMult> {

  ADMat          A;
  Vec            x_v;
  AdjointVecData x_i;
  AdjointVecData y_i;

  ADData_MatMult(ADMat A, ADVec x, ADVec y) : A(), x_v(), x_i(x), y_i(y) {
    PetscCallVoid(VecDuplicate(x->vec, &x_v));
    PetscCallVoid(VecCopy(x->vec, x_v));

    ADMatCopyForReverse(A, &this->A);
  }

  void reverse(Tape* tape, VectorInterface* vi) {
    Vec x_b;
    Vec y_b;
    Mat A_b;
    PetscCallVoid(x_i.createAdjoint(&x_b, 1));
    PetscCallVoid(y_i.createAdjoint(&y_b, 1));
    PetscCallVoid(MatDuplicate(A->mat, MAT_SHARE_NONZERO_PATTERN, &A_b));

    PetscInt low;
    PetscInt high;
    PetscCallVoid(VecGetOwnershipRange(y_b,&low, &high));


    // TODO: Refactor dyadic iteration procedure.
    std::vector<Real> remote_x_v(0);
    std::set<PetscInt> col_list= {};
    std::map<PetscInt, PetscInt> colmap;

    auto dyadic_init = [&] (PetscInt row, PetscInt col, Wrapper& value) {
      if( col < low || high <= col) {
        col_list.insert(col);
      }
    };
    PetscCallVoid(ADObjIterateAllEntries(A, dyadic_init));

    std::vector<PetscInt> col_vec(0);
    col_vec.reserve(col_list.size());
    remote_x_v.resize(col_list.size());
    col_vec.insert(col_vec.begin(), col_list.begin(), col_list.end());

    PetscInt pos = 0;
    for(PetscInt col : col_vec) {
      colmap[col] = pos;
      pos += 1;
    }

    int dim = vi->getVectorSize();

    PetscCallVoid(VecGetValuesNonLocal(x_v, col_vec.size(), col_vec.data(), remote_x_v.data()));

    int cur_dim = 0;
    auto dyadic_update = [&] (PetscInt row, PetscInt col, Wrapper& value) {
      Real entry_y_b;
      Real entry_x_v;
      PetscCallVoid(VecGetValues(y_b, 1, &row, &entry_y_b));

      if( low <= col && col < high) {
        PetscCallVoid(VecGetValues(x_v, 1, &col, &entry_x_v));
      } else {
        entry_x_v = remote_x_v[colmap[col]];
      }
      vi->updateAdjoint(value.getIdentifier(), cur_dim, entry_y_b * entry_x_v);

      value.value() = entry_y_b * entry_x_v;
    };

    for(; cur_dim < dim; cur_dim += 1) {
      y_i.getAdjoint(y_b, vi, cur_dim);

      PetscCallVoid(MatMultTranspose(A->mat, y_b, x_b));
      x_i.updateAdjoint(x_b, vi, cur_dim);

      PetscCallVoid(ADObjIterateAllEntries(A, dyadic_update));
    }

    PetscCallVoid(x_i.freeAdjoint(&x_b));
    PetscCallVoid(y_i.freeAdjoint(&y_b));
  }
};

PetscErrorCode MatMult(ADMat mat, ADVec x, ADVec y) {
  PetscCall(MatMult(mat->mat, x->vec, y->vec));

  ADData_MatMult* data = new ADData_MatMult(mat, x, y);
  data->push();

  return PETSC_SUCCESS;
}

struct ADData_MatNorm : public ReverseDataBase<ADData_MatNorm> {

  ADMat         x;
  NormType      type;
  Real          v_v;
  Identifier    v_i;

  ADData_MatNorm(ADMat x, NormType type, Number* v) : x(), type(type), v_v(v->getValue()), v_i() {

    Number::getTape().registerExternalFunctionOutput(*v);
    v_i = v->getIdentifier();

    ADMatCopyForReverse(x, &this->x);
  }

  void reverse(Tape* tape, VectorInterface* vi) {
    int dim = vi->getVectorSize();

    MPI_Comm comm;
    PetscObjectGetComm((PetscObject)x->mat, &comm);

    // Get the lhs adjoint.
    std::vector<Real> v_b(dim);
    vi->getAdjointVec(v_i, v_b.data());
    vi->resetAdjointVec(v_i);
    MPI_Allreduce(MPI_IN_PLACE, v_b.data(), dim, MPI_DOUBLE, MPI_SUM, comm);

    if(type == NORM_1 || type == NORM_INFINITY) {
      std::set<PetscInt> selected = {};

      // First select the rows/cols with the maximum values.
      if(type == NORM_1) {
        PetscInt col_size;
        PetscInt row_size;
        PetscCallVoid(MatGetSize(x->mat, &row_size, &col_size));


        std::vector<Real> col_sums(col_size);
        PetscCallVoid(MatGetColumnSumAbs(x->mat, col_sums.data()));

        PetscInt cur_col = 0;
        for(Real const& value_col : col_sums) {
          if(value_col == v_v) {
            selected.insert(cur_col);
          }
          cur_col += 1;
        }
      }
      else if(type == NORM_INFINITY) {
        PetscInt low;
        PetscInt high;
        PetscCallVoid(MatGetOwnershipRange(x->mat, &low, &high));
        PetscInt range = high - low;

        Vec row_sums;
        PetscCallVoid(MatCreateVecs(x->mat, &row_sums, NULL));
        PetscCallVoid(MatGetRowSumAbs(x->mat, row_sums));

        Real* row_sums_values;
        PetscCallVoid(VecGetArray(row_sums, &row_sums_values));

        for(PetscInt cur_row = 0; cur_row < range; cur_row += 1) {
          if(row_sums_values[cur_row] == v_v) {
            selected.insert(cur_row + low);
          }
        }

        PetscCallVoid(VecRestoreArray(row_sums, &row_sums_values));
        PetscCallVoid(VecDestroy(&row_sums));
      }

      // Iterate and perform the adjoint update.
      auto func = [&](PetscInt row, PetscInt col, Wrapper& value) {
        PetscInt sel = type == NORM_1 ? col : row;
        if(selected.end() == selected.find(sel)) {
          return;
        }

        Real jac = 0.0;
        if(value.getValue() < 0.0) {
          jac = -1.0;
        }
        else if(value.getValue() > 0.0) {
          jac = 1.0;
        }

        for(int i = 0; i < dim; i += 1) {
          vi->updateAdjoint(value.getIdentifier(), i, v_b[i] * jac);
        }
      };
      ADObjIterateAllEntries(x, func);
    }
    else if(type == NORM_FROBENIUS) {
      auto func = [&](PetscInt row, PetscInt col, Wrapper& value) {
        for(int i = 0; i < dim; i += 1) {
          vi->updateAdjoint(value.getIdentifier(), i, v_b[i] * value.value() / v_v);
        }
      };
      ADObjIterateAllEntries(x, func);
    }
    else {
      // TODO: Throw error
    }
  }
};

PetscErrorCode MatNorm(ADMat x, NormType type, Number *val) {
  PetscCall(MatNorm(x->mat, type, &val->value()));

  ADData_MatNorm* data = new ADData_MatNorm(x, type, val);
  data->push();

  return PETSC_SUCCESS;
}

PetscErrorCode MatSeqAIJSetPreallocation (ADMat B, PetscInt nz, const PetscInt nnz[]) {
  return MatSeqAIJSetPreallocation(B->mat, nz, nnz);
}

PetscErrorCode MatSetFromOptions(ADMat mat) {
  return MatSetFromOptions(mat->mat);
}

PetscErrorCode MatSetOption(ADMat x, MatOption op, PetscBool flag) {
  return MatSetOption(x->mat, op, flag);
}

PetscErrorCode MatSetSizes(ADMat mat, PetscInt m, PetscInt n, PetscInt M, PetscInt N) {
  PetscCall(MatSetSizes(mat->mat, m, n, M, N));

  return PETSC_SUCCESS;
}

PetscErrorCode MatSetValues(ADMat mat, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], const Number v[], InsertMode addv) {
  std::vector<Real> v_p(m * n);
  for(size_t i = 0; i < v_p.size(); i += 1) {
    v_p[i] = v[i].getValue();
  }
  PetscCall(MatSetValues(mat->mat, m, idxm, n, idxn, v_p.data(), addv));

  ADData_MatSetValues* data = nullptr;
  if(nullptr == mat->transaction_data) {
    data = new ADData_MatSetValues(addv);
    mat->transaction_data = data;
  }
  else {
    data = (ADData_MatSetValues*)mat->transaction_data;
  }

  data->addEntries(m, idxm, n, idxn, v);

  return PETSC_SUCCESS;
}

PetscErrorCode MatSetValue(ADMat mat, PetscInt i, PetscInt j, Number v, InsertMode addv) {
  return MatSetValues(mat, 1, &i, 1, &j, &v, addv);
}

PetscErrorCode MatView(ADMat mat, PetscViewer viewer) {
  return MatView(mat->mat, viewer);
}

// PetscErrorCode MatZeroEntries            (ADMat mat);


PetscErrorCode MatSeqAIJGetEntrySize(Mat mat, PetscInt* entries) {
  PetscInt const* row_offset;
  PetscInt low;
  PetscInt high;
  PetscInt row_range;

  PetscCall(MatGetOwnershipRange(mat, &low, &high));
  PetscCall(MatSeqAIJGetCSRAndMemType(mat, &row_offset, nullptr, nullptr, nullptr));

  row_range = high - low;
  *entries = row_offset[row_range];

  return PETSC_SUCCESS;
}

PetscErrorCode MatAIJGetEntrySize(Mat mat, PetscInt* diag_entries, PetscInt* off_diag_entries) {
  Mat matd;
  Mat mato;
  PetscInt const* colmap;
  PetscCall(MatMPIAIJGetSeqAIJ(mat, &matd, &mato, &colmap));

  PetscCall(MatSeqAIJGetEntrySize(matd, diag_entries));
  PetscCall(MatSeqAIJGetEntrySize(mato, off_diag_entries));

  return PETSC_SUCCESS;
}

void ADMatCreateADData(ADMat mat) {
  // TODO: Check for matrix type
  PetscInt diag_size;
  PetscInt off_diag_size;

  PetscCallVoid(MatAIJGetEntrySize(mat->mat, &diag_size, &off_diag_size));

  mat->ad_size = diag_size + off_diag_size;
  mat->ad_size_diag = diag_size;
  mat->ad_data = new Identifier[mat->ad_size];
  memset(mat->ad_data, 0, sizeof(Identifier) * mat->ad_size);
}

void ADMatCopyForReverse(ADMat mat, ADMat* newm) {
  *newm = new ADMatImpl();

  PetscCallVoid(MatDuplicate(mat->mat, MAT_COPY_VALUES, &(*newm)->mat));

  ADMatCreateADData(*newm);

  std::copy(mat->ad_data, &mat->ad_data[mat->ad_size], (*newm)->ad_data);
}

AP_NAMESPACE_END
