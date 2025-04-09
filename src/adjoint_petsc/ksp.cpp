#include <adjoint_petsc/ksp.h>

#include "util/addata_helper.hpp"
#include "util/mat_iterator_util.hpp"
#include "util/dyadic_product_helper.hpp"

AP_NAMESPACE_START

KSP& ADKSPImpl::getKSP() {
  return *ksp.get();
}

void deleteKSP(KSP* ksp) {
  PetscCallVoid(KSPDestroy(ksp));

  delete ksp;
}

/*-------------------------------------------------------------------------------------------------
 * PETSc functions
 */

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
  AdjointVecData      b_i;


  Vec                 x_v;
  DyadicProductHelper x_dyadic;
  AdjointVecData      x_i;

  Mat                 A_v;
  ADMatData*          A_i;
  Mat                 P_v;

  SharedKSP ksp;

  ADData_KSPSolve(ADKSP ksp, bool active_A, ADVec b, ADVec x) : b_i(b), x_v(), x_dyadic(), x_i(x), A_v(), A_i(), P_v(), ksp(ksp->ksp) {
    if(active_A) {
      ADMatCopyForReverse(ksp->Amat, &A_v, &A_i);

      PetscCallVoid(VecDuplicate(x->vec, &x_v));
      PetscCallVoid(VecCopy(x->vec, x_v));
      x_dyadic.init(A_v, x_v);
      x_dyadic.communicateValues(x_v);
    } else {
      ADMatCopyForReverse(ksp->Amat, &A_v, nullptr);
    }

    PetscObjectId id_A;
    PetscObjectId id_P;
    PetscObjectGetId((PetscObject)ksp->Amat->mat, &id_A);
    PetscObjectGetId((PetscObject)ksp->Pmat->mat, &id_P);
    if(id_A == id_P) {
      P_v = A_v;
    }
    else {
      PetscCallVoid(MatDuplicate(ksp->Pmat->mat, MAT_COPY_VALUES, &P_v));
    }
  }

  ~ADData_KSPSolve() {
    PetscCallVoid(MatDestroy(&A_v));

    if(nullptr != A_i) {
      delete A_i;
      PetscCallVoid(VecDestroy(&x_v));
    }
  }

  void reverse(Tape* tape, VectorInterface* vi) {
    PetscInt low;

    Vec x_b;
    Vec b_b;
    Mat A_b;

    PetscCallVoid(x_i.createAdjoint(&x_b, 1));
    PetscCallVoid(b_i.createAdjoint(&b_b, 1));
    PetscCallVoid(MatDuplicate(A_v, MAT_SHARE_NONZERO_PATTERN, &A_b));

    PetscCallVoid(VecGetOwnershipRange(b_b, &low, nullptr));

    PetscCallVoid(KSPSetOperators(*ksp.get(), A_v, P_v));

    int dim = vi->getVectorSize();

    int cur_dim = 0;
    auto dyadic_update = [&] (PetscInt row, PetscInt col, Real& value, Identifier& id) {
      (void) value;

      Real entry_b_b;
      Real entry_x_v;
      PetscCallVoid(VecGetValues(b_b, 1, &row, &entry_b_b));
      PetscCallVoid(x_dyadic.getValue(x_v, col, &entry_x_v));

      if( tape->isIdentifierActive(id)) {
        vi->updateAdjoint(id, cur_dim, -entry_b_b * entry_x_v);
      }
    };
    for(; cur_dim < dim; cur_dim += 1) {
      x_i.getAdjoint(x_b, vi, cur_dim);

      PetscCallVoid(KSPSolveTranspose(*ksp.get(), x_b, b_b));
      b_i.updateAdjoint(b_b, vi, cur_dim);

      if(A_i != nullptr) {
        PetscCallVoid(MatIterateAllEntries(dyadic_update, A_v, A_i));
      }
    }

    PetscCallVoid(x_i.freeAdjoint(&x_b));
    PetscCallVoid(b_i.freeAdjoint(&b_b));

    PetscCallVoid(MatDestroy(&A_b));
  }
};

PetscErrorCode KSPSolve(ADKSP ksp, ADVec b, ADVec x) {
  PetscCall(KSPSolve(ksp->getKSP(), b->vec, x->vec));

  bool active_A;
  bool active_b;
  ADMatIsActive(ksp->Amat, &active_A);
  ADVecIsActive(b, &active_b);

  if(active_b || active_A) {

    PetscCall(ADVecRegisterExternalFunctionOutput(x));
    ADData_KSPSolve* data = new ADData_KSPSolve(ksp, active_A, b, x);
    data->push();
  } else {
    PetscCall(ADVecMakePassive(x));
  }

  return PETSC_SUCCESS;
}

PetscErrorCode KSPView(ADKSP ksp, PetscViewer viewer) {
  // TODO: Implement AD version
  return KSPView(ksp->getKSP(), viewer);
}

AP_NAMESPACE_END
