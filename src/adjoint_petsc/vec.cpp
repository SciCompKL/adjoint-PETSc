#include <adjoint_petsc/vec.h>
#include <adjoint_petsc/util/petsc_missing.h>
#include <adjoint_petsc/util/addata_helper.hpp>

#include "util/vec_iterator_util.hpp"

AP_NAMESPACE_START

/*************************************************************
 * Helper functions
 */

PetscErrorCode createAdjointVec(Vec* vec, FuncCreate create, FuncInit init, PetscInt m, PetscInt M) {
  PetscCall(create(vec));
  PetscCall(init(*vec));
  PetscCall(VecSetSizes(*vec, m, M));
  PetscCall(VecSet(*vec,0.0));
  return PETSC_SUCCESS;
}

/*************************************************************
 * Vector functions
 */

PetscErrorCode VecAXPY(ADVec y, Number alpha, ADVec x) {
  // VecAXPY is purely local operation.
  Real* x_data;
  Real* y_data;

  PetscCall(VecGetArray(x->vec, &x_data));
  PetscCall(VecGetArray(y->vec, &y_data));

  for(PetscInt i = 0; i < x->ad_size; i += 1) {
    createRefType(y_data[i], y->ad_data[i]) += alpha * createRefType(x_data[i], x->ad_data[i]);
  }

  PetscCall(VecRestoreArray(x->vec, &x_data));
  PetscCall(VecRestoreArray(y->vec, &y_data));

  return PETSC_SUCCESS;
}

PetscErrorCode VecAYPX(ADVec y, Number beta, ADVec x) {
  // VecAYPX is purely local operation.
  Real* x_data;
  Real* y_data;

  PetscCall(VecGetArray(x->vec, &x_data));
  PetscCall(VecGetArray(y->vec, &y_data));

  for(PetscInt i = 0; i < x->ad_size; i += 1) {
    createRefType(y_data[i], y->ad_data[i]) = beta * createRefType(y_data[i], y->ad_data[i]) + createRefType(x_data[i], x->ad_data[i]);
  }

  PetscCall(VecRestoreArray(x->vec, &x_data));
  PetscCall(VecRestoreArray(y->vec, &y_data));

  return PETSC_SUCCESS;
}

struct ADData_SetValues {
  struct RhsData {
    PetscInt lhs_pos;
    Identifier rhs_identifier;
  };

  std::vector<Identifier> lhs_identifiers;
  std::vector<PetscInt> lhs_out_positions;

  std::vector<PetscInt> lhs_in_positions;
  std::vector<RhsData> rhs_data;

  std::vector<int> step_boundaries;

  InsertMode mode;

  Real* adjoint_data;
  int adjoint_step;


  FuncCreate createFunc;
  FuncInit   initFunc;
  PetscInt local_size;
  PetscInt global_size;

  ADData_SetValues(InsertMode m, ADVec vec) : lhs_in_positions(), rhs_data(), mode(m), adjoint_data(), adjoint_step(),
      createFunc(vec->createFunc), initFunc(vec->initFunc), local_size(), global_size() {

    PetscCallVoid(VecGetLocalSize(vec->vec, & local_size));
    PetscCallVoid(VecGetSize(vec->vec, & global_size));
  }

  void addEntries(PetscInt ssize, PetscInt const* list, Number const* values) {
    step_boundaries.push_back((int)rhs_data.size());
    for (int i = 0; i < ssize; i += 1) {
      if(list[i] < 0) {
        continue;
      }
      addEntry(list[i], values[i]);
    }

    if(Number::getTape().isActive()) {
      Number::getTape().pushExternalFunction(codi::ExternalFunction<Tape>::create(&step_reverse, this, &no_delete));
    }
  }

  void finalize(ADVec vec) {
    step_boundaries.push_back((int)rhs_data.size());

    // Compute the activity for this operation.
    Vec activity_vector;
    VecDuplicate(vec->vec, &activity_vector);

    VecSet(activity_vector, 0.0);
    std::vector<Real> values(lhs_in_positions.size(), 1.0);
    VecSetValues(activity_vector, (int)lhs_in_positions.size(), lhs_in_positions.data(), values.data(), ADD_VALUES);
    VecAssemblyBegin(activity_vector);
    VecAssemblyEnd(activity_vector);

    int local_size;
    VecGetLocalSize(activity_vector, &local_size);

    Real* activity_vector_data;
    Real* vec_data;
    VecGetArray(activity_vector, &activity_vector_data);
    VecGetArray(vec->vec, &vec_data);

    Tape& tape = Number::getTape();
    for(int i = 0; i < local_size; i += 1) {
      if(0 != activity_vector_data[i]) {

        Wrapper temp = createRefType(vec_data[i], vec->ad_data[i]);
        if(tape.isActive()) {

          tape.registerExternalFunctionOutput(temp);

          lhs_out_positions.push_back(i);
          lhs_identifiers.push_back(vec->ad_data[i]);
        } else {
          tape.deactivateValue(temp);
        }
      }
    }

    VecRestoreArray(activity_vector, &activity_vector_data);
    VecRestoreArray(vec->vec, &vec_data);

    if(tape.isActive()) {
      Number::getTape().pushExternalFunction(codi::ExternalFunction<Tape>::create(&assemble_reverse, this, &ad_delete));
    } else {
      delete this;
    }

    VecDestroy(&activity_vector);
  }

  static void step_reverse(Tape* tape, void* d, VectorInterface* va) {
    ADData_SetValues* data = (ADData_SetValues*)d;
    int ad_vec_size = va->getVectorSize();

    data->adjoint_step -= 1;
    int from = data->step_boundaries[data->adjoint_step];
    int to = data->step_boundaries[data->adjoint_step + 1];

    for(int i = from; i < to; i += 1) {
      va->updateAdjointVec(data->rhs_data[i].rhs_identifier, &data->adjoint_data[data->rhs_data[i].lhs_pos * ad_vec_size]);
    }

    if(data->adjoint_step == 0) {
      delete [] (data->adjoint_data);
    }
  }

  static void assemble_reverse(Tape* tape, void* d, VectorInterface* va) {
    ADData_SetValues* data = (ADData_SetValues*)d;
    int ad_vec_size = va->getVectorSize();

    data->adjoint_step = (int)data->step_boundaries.size() - 1;

    Vec adjoint_vec;
    createAdjointVec(&adjoint_vec, data->createFunc, data->initFunc, data->local_size * ad_vec_size, data->global_size * ad_vec_size);

    Real* local_data;
    VecGetArray(adjoint_vec, &local_data);

    for(int i = 0; i < (int)data->lhs_identifiers.size(); i += 1) {
      va->getAdjointVec(data->lhs_identifiers[i], &local_data[data->lhs_out_positions[i] * ad_vec_size]);
      if(INSERT_VALUES == data->mode) {
        va->resetAdjointVec(data->lhs_identifiers[i]);
      }
    }

    VecRestoreArray(adjoint_vec, &local_data);

    data->adjoint_data = new Real[ad_vec_size * data->lhs_in_positions.size()];
    if(ad_vec_size == 1) {
      VecGetValuesNonLocal(adjoint_vec, (int)data->lhs_in_positions.size(), data->lhs_in_positions.data(), data->adjoint_data);
    } else {
      std::vector<PetscInt> positions(data->lhs_in_positions.size() * ad_vec_size);
      for(int i = 0; i < (int)data->lhs_in_positions.size(); i += 1) {
        for(int dim = 0; dim < ad_vec_size; dim += 1) {
          positions[i * ad_vec_size + dim] = data->lhs_in_positions[i] + dim;
        }
      }

      VecGetValues(adjoint_vec, (int)positions.size(), positions.data(), data->adjoint_data);
    }

    VecDestroy(&adjoint_vec);
  }

  static void no_delete(Tape* tape, void* data) {
    (void)tape;
    (void)data;
  }

  static void ad_delete(Tape* tape, void* d) {
    (void)tape;
    ADData_SetValues* data = (ADData_SetValues*)d;

    delete data;
  }


  private:
  void addEntry(PetscInt lhs_pos, Number const& rhs_value) {
    PetscInt lhs_cache_pos = insertLhsPos(lhs_pos);

    rhs_data.push_back({lhs_cache_pos, rhs_value.getIdentifier()});
  }

  PetscInt insertLhsPos(PetscInt lhs_pos) {
    // Search for the position.
    for(size_t i = 0; i < lhs_in_positions.size(); i += 1) {
      if(lhs_pos == lhs_in_positions[i]) {
        return i;
      }
    }

    // Did not find the position add it.
    lhs_in_positions.push_back(lhs_pos);

    return (PetscInt)lhs_in_positions.size() - 1;
  }
};

PetscErrorCode VecAssemblyBegin(ADVec vec) {
  if (nullptr != vec) {
    return VecAssemblyBegin(vec->vec);
  }
  else {
    return PETSC_SUCCESS;
  }
}

PetscErrorCode VecAssemblyEnd  (ADVec vec) {
  if (nullptr != vec) {
    PetscErrorCode r = VecAssemblyEnd(vec->vec);

    if(nullptr != vec->transaction_data) {
      ADData_SetValues* data = reinterpret_cast<ADData_SetValues*>(vec->transaction_data);

      data->finalize(vec);
      vec->transaction_data = nullptr;
    }

    return r;
  }
  else {
    return PETSC_SUCCESS;
  }
}

PetscErrorCode VecCopy(ADVec x, ADVec y) {
  Real* x_data;
  Real* y_data;

  PetscCall(VecGetArray(x->vec, &x_data));
  PetscCall(VecGetArray(y->vec, &y_data));

  for(PetscInt i = 0; i < x->ad_size; i += 1) {
    createRefType(y_data[i], y->ad_data[i]) = createRefType(x_data[i], x->ad_data[i]);
  }

  PetscCall(VecRestoreArray(x->vec, &x_data));
  PetscCall(VecRestoreArray(y->vec, &y_data));

  return PETSC_SUCCESS;
}

PetscErrorCode VecCreate(MPI_Comm comm, ADVec* vec) {
  *vec = new ADVecImpl();

  PetscCall(VecCreate(comm, &(*vec)->vec));
  (*vec)->createFunc = [comm] (Vec* v) -> PetscErrorCode {
    PetscCall(VecCreate(comm, v)); return PETSC_SUCCESS;
  };

  return PETSC_SUCCESS;
}

PetscErrorCode VecDestroy(ADVec* vec) {
  Real* vec_data;

  PetscCall(VecGetArray((*vec)->vec, &vec_data));

  Tape& tape = Number::getTape();
  for(PetscInt i = 0; i < (*vec)->ad_size; i += 1) {
    Wrapper temp = createRefType(vec_data[i], (*vec)->ad_data[i]);
    tape.deactivateValue(temp);
  }

  PetscCall(VecRestoreArray((*vec)->vec, &vec_data));

  delete [] (*vec)->ad_data;

  PetscCall(VecDestroy(&(*vec)->vec));

  delete *vec;

  return PETSC_SUCCESS;
}

struct ADData_VecDot : public ReverseDataBase<ADData_VecDot> {
  MPI_Comm comm;
  Identifier val_i;

  std::vector<Identifier> x_i;
  std::vector<Real> x_v;

  std::vector<Identifier> y_i;
  std::vector<Real> y_v;

  ADData_VecDot(ADVec x, ADVec y, Number* val) : comm(MPI_COMM_NULL), val_i(val->getIdentifier()), x_i(x->ad_size),
      x_v(x->ad_size), y_i(x->ad_size), y_v(x->ad_size) {
    PetscCallVoid(PetscObjectGetComm((PetscObject)x->vec, &comm));
    PetscCallVoid(AdjointVecData::extractIdentifier(x, x_i.data()));
    PetscCallVoid(AdjointVecData::extractPrimal(x, x_v.data()));
    PetscCallVoid(AdjointVecData::extractIdentifier(y, y_i.data()));
    PetscCallVoid(AdjointVecData::extractPrimal(y, y_v.data()));
  }

  void reverse(Tape* tape, VectorInterface* vi) {
    (void)tape;

    int dim = vi->getVectorSize();

    std::vector<Real> adj(dim);

    vi->getAdjointVec(val_i, adj.data());
    vi->resetAdjointVec(val_i);

    MPI_Allreduce(MPI_IN_PLACE, adj.data(), dim, MPI_DOUBLE, MPI_SUM, comm);

    for(size_t i = 0; i < x_i.size(); i += 1) {
      for(int d = 0; d < dim; d += 1) {
        vi->updateAdjoint(x_i[i], d, y_v[i] * adj[d]);
        vi->updateAdjoint(y_i[i], d, x_v[i] * adj[d]);
      }
    }
  }
};

PetscErrorCode VecDot(ADVec x, ADVec y, Number* val) {
  PetscCall(VecDot(x->vec, y->vec, &val->value()));

  Number::getTape().registerExternalFunctionOutput(*val);
  ADData_VecDot* data = new ADData_VecDot(x, y, val);
  data->push();

  return PETSC_SUCCESS;
}

PetscErrorCode VecDuplicate(ADVec vec, ADVec* newv) {
  *newv = new ADVecImpl();
  PetscCall(VecDuplicate(vec->vec, &(*newv)->vec));

  ADVecCreateADData(*newv);
  (*newv)->createFunc = vec->createFunc;
  (*newv)->initFunc   = vec->initFunc;

  return PETSC_SUCCESS;
}

PetscErrorCode VecGetArray(ADVec vec, WrapperArray* a) {
  Real* primals;
  PetscCall(VecGetArray(vec->vec, &primals));
  *a = WrapperArray(primals, vec->ad_data);

  return PETSC_SUCCESS;
}

PetscErrorCode VecGetLocalSize(ADVec vec, PetscInt* size) {
  return VecGetLocalSize(vec->vec, size);
}

PetscErrorCode VecGetOwnershipRange(ADVec x, PetscInt *low, PetscInt *high) {
  return VecGetOwnershipRange(x->vec, low, high);
}

PetscErrorCode VecGetSize(ADVec vec, PetscInt* size) {
  return VecGetSize(vec->vec, size);
}

PetscErrorCode VecGetValues(ADVec x, PetscInt ni, PetscInt const* ix, Number* y) {
  Real* values;
  PetscCall(VecGetArray(x->vec, &values));

  for(PetscInt i = 0; i < ni; i += 1) {
    y[i] = createRefType(values[ix[i]], x->ad_data[ix[i]]);
  }

  PetscCall(VecRestoreArray(x->vec, &values));

  return PETSC_SUCCESS;
}

struct ADData_VecMax : public ReverseDataBase<ADData_VecMax> {
  MPI_Comm comm;
  Identifier val_i;

  std::vector<Identifier> x_i; // All values that are the maximum. (Only one if p was set.

  ADData_VecMax(ADVec x, PetscInt* p, Number* val) : comm(MPI_COMM_NULL), val_i(val->getIdentifier()), x_i(0) {
    PetscCallVoid(PetscObjectGetComm((PetscObject)x->vec, &comm));

    if(nullptr != p) {
      // Only update the specific value indicated by p.
      PetscInt low;
      PetscInt hight;
      PetscCallVoid(VecGetOwnershipRange(x->vec, &low, &hight));

      if(low <= *p && *p < hight) {
        x_i.push_back(x->ad_data[*p - low]);
      }
    }
    else {
      // Update all values that are the same as max.
      Real* x_data;

      PetscCallVoid(VecGetArray(x->vec, &x_data));
      for(int i = 0; i < x->ad_size; i += 1) {
        if(val->getValue() == x_data[i]) {
          x_i.push_back(x->ad_data[i]);
        }
      }
      PetscCallVoid(VecRestoreArray(x->vec, &x_data));
    }
  }

  void reverse(Tape* tape, VectorInterface* vi) {
    (void)tape;

    int dim = vi->getVectorSize();

    std::vector<Real> adj(dim);

    vi->getAdjointVec(val_i, adj.data());
    vi->resetAdjointVec(val_i);

    MPI_Allreduce(MPI_IN_PLACE, adj.data(), dim, MPI_DOUBLE, MPI_SUM, comm);

    for(size_t i = 0; i < x_i.size(); i += 1) {
      vi->updateAdjointVec(x_i[i], adj.data());
    }
  }
};

PetscErrorCode VecMax(ADVec x, PetscInt* p, Number* val) {
  PetscCall(VecMax(x->vec, p, &val->value()));

  Number::getTape().registerExternalFunctionOutput(*val);
  ADData_VecMax* data = new ADData_VecMax(x, p, val);
  data->push();

  return PETSC_SUCCESS;
}

struct ADData_VecNorm : public ReverseDataBase<ADData_VecNorm> {
  MPI_Comm comm;
  std::array<Identifier, 2> val_i;
  Real val_v; // Only used for max.

  NormType type;

  std::vector<Real>       x_v;
  std::vector<Identifier> x_i;

  ADData_VecNorm(ADVec x, NormType type, Number* val) : comm(MPI_COMM_NULL), val_i(), val_v(val->getValue()),
      type(type), x_v(x->ad_size), x_i(x->ad_size) {
    PetscCallVoid(PetscObjectGetComm((PetscObject)x->vec, &comm));


    int size = (type == NORM_1_AND_2) ? 2 : 1;
    for(int i = 0; i < size; i += 1) {
      val_i[i] = val[i].getIdentifier();
    }

    PetscCallVoid(AdjointVecData::extractPrimal(x, x_v.data()));
    PetscCallVoid(AdjointVecData::extractIdentifier(x, x_i.data()));
  }

  void reverse(Tape* tape, VectorInterface* vi) {
    (void)tape;

    int dim = vi->getVectorSize();
    int size = (type == NORM_1_AND_2) ? 2 : 1;

    std::vector<Real> adj(dim * size);

    for(int i = 0; i < size; i += 1) {
      vi->getAdjointVec(val_i[i], adj.data() + i * dim);
      vi->resetAdjointVec(val_i[i]);
    }

    MPI_Allreduce(MPI_IN_PLACE, adj.data(), dim * size, MPI_DOUBLE, MPI_SUM, comm);

    if(type == NORM_1 || type == NORM_1_AND_2) {
      for(size_t i = 0; i < x_i.size(); i += 1) {
        for(int d = 0; d < dim; d += 1) {
          if(x_v[i] < 0.0) { vi->updateAdjoint(x_i[i], d, -adj[d]); }
          else if(x_v[i] > 0.0) { vi->updateAdjoint(x_i[i], d, adj[d]); }
        }
      }
    }

    if(type == NORM_2 || type == NORM_1_AND_2|| type == NORM_FROBENIUS) {
      int pos = (type == NORM_1_AND_2) ? 1 : 0;

      Real* adj_offset = adj.data() + pos * dim;

      for(size_t i = 0; i < x_i.size(); i += 1) {
        Real jac = 2.0 * x_v[i];
        for(int d = 0; d < dim; d += 1) {
          vi->updateAdjoint(x_i[i], d, jac * adj_offset[d]);
        }
      }
    }
    else if(type == NORM_MAX) {
      for(size_t i = 0; i < x_i.size(); i += 1) {
        if(abs(x_v[i]) == val_v) {
          for(int d = 0; d < dim; d += 1) {
            if(x_v[i] < 0.0) { vi->updateAdjoint(x_i[i], d, -adj[d]); }
            else if(x_v[i] > 0.0) { vi->updateAdjoint(x_i[i], d, adj[d]); }
          }
        }
      }
    }
    else {
      // TODO: throw error.
    }
  }
};

PetscErrorCode VecNorm(ADVec x, NormType type, Number* val) {
  std::array<Real, 2> val_p;

  PetscCall(VecNorm(x->vec, type, val_p.data()));

  int size = (type == NORM_1_AND_2) ? 2 : 1;
  for(int i = 0; i < size; i += 1) {
    val[i].setValue(val_p[i]);
    Number::getTape().registerExternalFunctionOutput(val[i]);
  }

  ADData_VecNorm* data = new ADData_VecNorm(x, type, val);
  data->push();

  return PETSC_SUCCESS;
}

PetscErrorCode VecPointwiseDivide(ADVec w, ADVec x, ADVec y) {
  // VecPointwiseDivide is purely local operation.
  Real* w_data;
  Real* x_data;
  Real* y_data;

  PetscCall(VecGetArray(w->vec, &w_data));
  PetscCall(VecGetArray(x->vec, &x_data));
  PetscCall(VecGetArray(y->vec, &y_data));

  for(PetscInt i = 0; i < x->ad_size; i += 1) {
    createRefType(w_data[i], w->ad_data[i]) = createRefType(x_data[i], x->ad_data[i]) / createRefType(y_data[i], y->ad_data[i]);
  }

  PetscCall(VecRestoreArray(w->vec, &x_data));
  PetscCall(VecRestoreArray(x->vec, &x_data));
  PetscCall(VecRestoreArray(y->vec, &y_data));

  return PETSC_SUCCESS;
}

PetscErrorCode VecPointwiseMult (ADVec w, ADVec x, ADVec y) {
  // VecPointwiseMult is purely local operation.
  Real* w_data;
  Real* x_data;
  Real* y_data;

  PetscCall(VecGetArray(w->vec, &w_data));
  PetscCall(VecGetArray(x->vec, &x_data));
  PetscCall(VecGetArray(y->vec, &y_data));

  for(PetscInt i = 0; i < x->ad_size; i += 1) {
    createRefType(w_data[i], w->ad_data[i]) = createRefType(x_data[i], x->ad_data[i]) * createRefType(y_data[i], y->ad_data[i]);
  }

  PetscCall(VecRestoreArray(w->vec, &x_data));
  PetscCall(VecRestoreArray(x->vec, &x_data));
  PetscCall(VecRestoreArray(y->vec, &y_data));

  return PETSC_SUCCESS;
}

PetscErrorCode VecPow(ADVec x, Number p) {
  // VecPow is purely local operation.
  Real* x_data;

  PetscCall(VecGetArray(x->vec, &x_data));

  for(PetscInt i = 0; i < x->ad_size; i += 1) {
    createRefType(x_data[i], x->ad_data[i]) = pow(createRefType(x_data[i], x->ad_data[i]), p);
  }

  PetscCall(VecRestoreArray(x->vec, &x_data));

  return PETSC_SUCCESS;
}

PetscErrorCode VecRestoreArray  (ADVec vec, WrapperArray* a) {
  Real* primals = a->getValues();
  PetscCall(VecRestoreArray(vec->vec, &primals));

  return PETSC_SUCCESS;
}


PetscErrorCode VecScale(ADVec x, Number alpha) {
  // VecScale is purely local operation.
  Real* x_data;

  PetscCall(VecGetArray(x->vec, &x_data));

  for(PetscInt i = 0; i < x->ad_size; i += 1) {
    createRefType(x_data[i], x->ad_data[i]) *= alpha;
  }

  PetscCall(VecRestoreArray(x->vec, &x_data));

  return PETSC_SUCCESS;
}

PetscErrorCode VecSet(ADVec x, Number alpha) {
  // VecSet is purely local operation.
  Real* x_data;

  PetscCall(VecGetArray(x->vec, &x_data));

  for(PetscInt i = 0; i < x->ad_size; i += 1) {
    createRefType(x_data[i], x->ad_data[i]) = alpha;
  }

  PetscCall(VecRestoreArray(x->vec, &x_data));

  return PETSC_SUCCESS;
}

PetscErrorCode VecSetFromOptions(ADVec vec) {
  PetscCall(VecSetFromOptions(vec->vec));
  vec->initFunc = [] (Vec v) -> PetscErrorCode {
    PetscCall(VecSetFromOptions(v)); return PETSC_SUCCESS;
  };

  ADVecCreateADData(vec);

  return PETSC_SUCCESS;
}

PetscErrorCode VecSetOption(ADVec x, VecOption op, PetscBool flag) {
  return VecSetOption(x->vec, op, flag);
}

PetscErrorCode VecSetSizes(ADVec vec, PetscInt m, PetscInt M) {
  PetscCall(VecSetSizes(vec->vec, m, M));

  return PETSC_SUCCESS;
}

PetscErrorCode VecSetType(ADVec vec, VecType newType) {
  PetscCall(VecSetType(vec->vec, newType));
  vec->initFunc = [newType] (Vec v) -> PetscErrorCode {
    PetscCall(VecSetType(v, newType)); return PETSC_SUCCESS;
  };

  ADVecCreateADData(vec);

  return PETSC_SUCCESS;
}

PetscErrorCode VecSetValue(ADVec vec, PetscInt i, Number y, InsertMode iora) {
  return VecSetValues(vec, 1, &i, &y, iora);
}

PetscErrorCode VecSetValues(ADVec vec, PetscInt ni, PetscInt const* ix, Number const* y, InsertMode iora) {

  std::vector<Real> y_conv(ni);
  for(PetscInt i = 0; i < ni; i += 1) {
    y_conv[i] = y[i].getValue();
  }

  PetscErrorCode r = VecSetValues(vec->vec, ni, ix, y_conv.data(), iora);

  ADData_SetValues* data = nullptr;
  if(nullptr == vec->transaction_data) {
    data = new ADData_SetValues(iora, vec);
    vec->transaction_data = data;
  }
  else {
    data = (ADData_SetValues*)vec->transaction_data;
  }

  if(data->mode != iora) {
    std::cerr << "Different insert mode" << std::endl;
  }

  data->addEntries(ni, ix, y);

  return r;
}

PetscErrorCode VecShift(ADVec x, Number shift) {
  // VecShift is purely local operation.
  Real* x_data;

  PetscCall(VecGetArray(x->vec, &x_data));

  for(PetscInt i = 0; i < x->ad_size; i += 1) {
    createRefType(x_data[i], x->ad_data[i]) += shift;
  }

  PetscCall(VecRestoreArray(x->vec, &x_data));

  return PETSC_SUCCESS;
}

struct ADData_VecSum : public ReverseDataBase<ADData_VecSum> {
  MPI_Comm comm;
  Identifier val_i;

  std::vector<Identifier> x_i;

  ADData_VecSum(ADVec x, Number* val) : comm(MPI_COMM_NULL), val_i(val->getIdentifier()), x_i(x->ad_size) {
    PetscCallVoid(PetscObjectGetComm((PetscObject)x->vec, &comm));
    PetscCallVoid(AdjointVecData::extractIdentifier(x, x_i.data()));
  }

  void reverse(Tape* tape, VectorInterface* vi) {
    (void)tape;

    int dim = vi->getVectorSize();

    std::vector<Real> adj(dim);

    vi->getAdjointVec(val_i, adj.data());
    vi->resetAdjointVec(val_i);

    MPI_Allreduce(MPI_IN_PLACE, adj.data(), dim, MPI_DOUBLE, MPI_SUM, comm);

    for(size_t i = 0; i < x_i.size(); i += 1) {
      vi->updateAdjointVec(x_i[i], adj.data());
    }
  }
};

PetscErrorCode VecSum(ADVec x, Number* sum) {
  PetscCall(VecSum(x->vec, &sum->value()));

  Number::getTape().registerExternalFunctionOutput(*sum);
  ADData_VecSum* data = new ADData_VecSum(x, sum);
  data->push();

  return PETSC_SUCCESS;
}

PetscErrorCode VecView(ADVec vec, PetscViewer viewer) {
  // TODO: Maybe add special AD output options.
  auto func = [&](PetscInt row, Wrapper& value) {
    (void)row;
    //std::cout << value.getValue() << "\n";
    std::cout << value.getIdentifier() << " " << value.getValue() << "\n";
  };
  std::cout.setf(std::ios::scientific);
  std::cout.setf(std::ios::showpos);
  std::cout.precision(12);
  std::cout << "Vector of global size M=" << vec->ad_size << std::endl; // TODO:
  std::cout << "cpu +0 #rows: " << vec->ad_size << std::endl; // TODO:
  VecIterateAllEntries(func, vec);
  std::cout.flush();

  return PETSC_SUCCESS;
}

PetscErrorCode VecWAXPY(ADVec w, Number alpha, ADVec x, ADVec y) {
  // VecWAXPY is purely local operation.
  Real* w_data;
  Real* x_data;
  Real* y_data;

  PetscCall(VecGetArray(w->vec, &w_data));
  PetscCall(VecGetArray(x->vec, &x_data));
  PetscCall(VecGetArray(y->vec, &y_data));

  for(PetscInt i = 0; i < x->ad_size; i += 1) {
    createRefType(w_data[i], w->ad_data[i]) = alpha * createRefType(x_data[i], x->ad_data[i]) + createRefType(y_data[i], y->ad_data[i]);
  }

  PetscCall(VecRestoreArray(w->vec, &w_data));
  PetscCall(VecRestoreArray(x->vec, &x_data));
  PetscCall(VecRestoreArray(y->vec, &y_data));

  return PETSC_SUCCESS;
}

/*************************************************************
 * ADVector functions
 */

void ADVecCreateADData(ADVec vec) {

  if(nullptr != vec->vec) {

    VecGetLocalSize(vec->vec, &vec->ad_size);

    vec->ad_data = new Identifier[vec->ad_size];
    memset(vec->ad_data, 0, vec->ad_size * sizeof(Identifier));
  }
  else {
    vec->ad_data = nullptr;
    vec->ad_size = 0;
  }
}

struct ADData_ViewReverse : public ReverseDataBase<ADData_ViewReverse> {

  AdjointVecData vec_i;
  std::string m;
  int id;
  PetscViewer viewer;

  ADData_ViewReverse(ADVec vec, std::string m, int id, PetscViewer viewer) : vec_i(vec), m(m), id(id), viewer(viewer) {}


  void reverse(Tape* tape, VectorInterface* vi) {
    Vec vec_b;
    PetscCallVoid(vec_i.createAdjoint(&vec_b, 1));

    PetscInt low;
    PetscCallVoid(VecGetOwnershipRange(vec_b, &low, nullptr));

    int dim = vi->getVectorSize();

    auto func = [&](PetscInt row, Real& value) {
      //std::cout << value << "\n";
      std::cout << vec_i.ids[row - low] << " " << value << "\n";
    };
    std::cout.setf(std::ios::scientific);
    std::cout.setf(std::ios::showpos);
    std::cout.precision(12);
    std::cout << m << " reverse id: " << id << std::endl;

    std::cout << "Vector of global size M=" << vec_i.ids.size() << std::endl; // TODO:
    std::cout << "cpu +0 #rows: " << vec_i.ids.size() << std::endl; // TODO:
    for(int cur_dim = 0; cur_dim < dim; cur_dim += 1) {
      vec_i.getAdjointNoReset(vec_b, vi, cur_dim);
      VecIterateAllEntries(func, vec_b);
      std::cout.flush();
    }

    PetscCallVoid(vec_i.freeAdjoint(&vec_b));
  }
};

void ADVecViewReverse(ADVec vec, std::string m, int id, PetscViewer viewer) {
  ADData_ViewReverse* data = new ADData_ViewReverse(vec, m, id, viewer);
  data->push();
}
void ADVecViewReverse(Vec vec, std::string m, int id, PetscViewer viewer) {
  (void)vec;
  (void)m;
  (void)id;
  (void)viewer;
}

AP_NAMESPACE_END
