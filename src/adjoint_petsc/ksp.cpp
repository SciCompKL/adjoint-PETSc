#include "../../include/adjoint_petsc/ksp.h"

#include "../../include/adjoint_petsc/util/addata_helper.hpp"
#include "../../include/adjoint_petsc/util/mat_iterator_util.hpp"
#include "../../include/adjoint_petsc/util/dyadic_product_helper.hpp"

AP_NAMESPACE_START

    KSP& ADKSPImpl::getKSP() {
      return *ksp.get();
    }

void deleteKSP(KSP* ksp) {
  PetscCallVoid(KSPDestroy(ksp));

  delete ksp;
}

PetscErrorCode KSPCreate(MPI_Comm comm, ADKSP *inksp) {
  *inksp = new ADKSPImpl();

  (*inksp)->ksp = SharedKSP(new KSP(), deleteKSP);
  PetscCall(KSPCreate(comm, &(*inksp)->getKSP()));

  return PETSC_SUCCESS;
}

PetscErrorCode KSPDestroy(ADKSP *ksp) {
  // Destroy is called by the smart pointer if necessary.

  delete *ksp;

  return PETSC_SUCCESS;
}

PetscErrorCode KSPGetIterationNumber(ADKSP ksp, PetscInt *its) {
  return KSPGetIterationNumber(ksp->getKSP(), its);
}

PetscErrorCode KSPGetPC(ADKSP ksp, PC *pc) {
  return KSPGetPC(ksp->getKSP(), pc);
}

PetscErrorCode KSPSetFromOptions(ADKSP ksp) {
  return KSPSetFromOptions(ksp->getKSP());
}

PetscErrorCode KSPSetInitialGuessNonzero(ADKSP ksp, PetscBool flg) {
  return KSPSetInitialGuessNonzero(ksp->getKSP(), flg);
}

PetscErrorCode KSPSetOperators(ADKSP ksp, ADMat Amat, ADMat Pmat) {
  PetscCall(KSPSetOperators(ksp->getKSP(), Amat->mat, Pmat->mat));

  ksp->Amat = Amat; // Store matrix here. It is handled in the solve routine.
  ksp->Pmat = Pmat; // Store matrix here. It is handled in the solve routine.

  return PETSC_SUCCESS;
}

struct ADData_KSPSolve : public ReverseDataBase<ADData_KSPSolve> {
  AdjointVecData    b_i;

  std::vector<Real> x_v;
  AdjointVecData    x_i;

  ADMat             A;
  Mat               P_v;

  SharedKSP ksp;

  ADData_KSPSolve(ADKSP ksp, ADVec b, ADVec x) : b_i(b), x_v(0), x_i(x), A(), P_v(), ksp(ksp->ksp) {
    x_v.resize(x_i.ids.size());
    PetscCallVoid(AdjointVecData::extractPrimal(x, x_v.data()));

    ADMatCopyForReverse(ksp->Amat, &A);

    PetscObjectId id_A;
    PetscObjectId id_P;
    PetscObjectGetId((PetscObject)ksp->Amat->mat, &id_A);
    PetscObjectGetId((PetscObject)ksp->Pmat->mat, &id_P);
    if(id_A == id_P) {
      P_v = A->mat;
    }
    else {
      PetscCallVoid(MatDuplicate(ksp->Pmat->mat, MAT_COPY_VALUES, &P_v));
    }
  }

  // TODO: Properly delete stuff

  void reverse(Tape* tape, VectorInterface* vi) {
    PetscInt low;

    Vec x_b;
    Vec b_b;
    Mat A_b;

    PetscCallVoid(x_i.createAdjoint(&x_b, 1));
    PetscCallVoid(b_i.createAdjoint(&b_b, 1));
    PetscCallVoid(MatDuplicate(A->mat, MAT_SHARE_NONZERO_PATTERN, &A_b));

    PetscCallVoid(VecGetOwnershipRange(b_b, &low, nullptr));

    PetscCallVoid(KSPSetOperators(*ksp.get(), A->mat, P_v));

    DyadicProductHelper dyadic = {};
    dyadic.init(A->mat, b_b);

    int dim = vi->getVectorSize();

    int cur_dim = 0;
    auto dyadic_update = [&] (PetscInt row, PetscInt col, Wrapper& value) {
      Real entry_x_v;
      Real entry_b_b;
      entry_x_v = x_v[row - low];
      PetscCallVoid(dyadic.getValue(b_b, col, &entry_b_b));

      vi->updateAdjoint(value.getIdentifier(), cur_dim, -entry_x_v * entry_b_b);
    };
    for(; cur_dim < dim; cur_dim += 1) {
      x_i.getAdjoint(x_b, vi, cur_dim);

      PetscCallVoid(KSPSolveTranspose(*ksp.get(), x_b, b_b));
      b_i.updateAdjoint(b_b, vi, cur_dim);

      dyadic.communicateValues(b_b);
      PetscCallVoid(PetscObjectIterateAllEntries(dyadic_update, A));
    }

    PetscCallVoid(x_i.freeAdjoint(&x_b));
    PetscCallVoid(b_i.freeAdjoint(&b_b));

    PetscCallVoid(MatDestroy(&A_b));
  }
};

PetscErrorCode KSPSolve(ADKSP ksp, ADVec b, ADVec x) {
  PetscCall(KSPSolve(ksp->getKSP(), b->vec, x->vec));

  PetscCall(AdjointVecData::registerExternalFunctionOutput(x));
  ADData_KSPSolve* data = new ADData_KSPSolve(ksp, b, x);
  data->push();

  return PETSC_SUCCESS;
}

PetscErrorCode KSPView(ADKSP ksp, PetscViewer viewer) {
  // TODO: Implement AD version
  return KSPView(ksp->getKSP(), viewer);
}

AP_NAMESPACE_END
