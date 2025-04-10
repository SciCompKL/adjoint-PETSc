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

#include <petscmat.h>

#include "vec.h"
#include "util/base.hpp"
#include "util/codi_def.hpp"

AP_NAMESPACE_START

struct ADMatData {

  int type;

  ADMatData(int type);
  ADMatData(ADMatData const&) = default;

  virtual ~ADMatData() {};
  virtual ADMatData* clone() = 0;

  void checkType(int castType);
};

struct ADMatImpl {
  Mat mat;

  ADMatData* mat_i;

  void* transaction_data;
};

using ADMat = ADMatImpl*;

/*-------------------------------------------------------------------------------------------------
 * PETSc functions
 */

PetscErrorCode MatAssemblyBegin          (ADMat mat, MatAssemblyType type);
PetscErrorCode MatAssemblyEnd            (ADMat mat, MatAssemblyType type);
PetscErrorCode MatConvert                (ADMat mat, MatType newtype, MatReuse reuse, ADMat *M);
PetscErrorCode MatCreate                 (MPI_Comm comm, ADMat* mat);
PetscErrorCode MatCreateAIJ              (MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], ADMat *A);
PetscErrorCode MatCreateDense            (MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Number *data, ADMat *A);
PetscErrorCode MatDestroy                (ADMat* mat);
PetscErrorCode MatDuplicate              (ADMat mat, MatDuplicateOption op, ADMat* newv);
PetscErrorCode MatGetInfo                (ADMat mat, MatInfoType flag, MatInfo *info);
PetscErrorCode MatGetLocalSize           (ADMat mat, PetscInt *m, PetscInt *n);
PetscErrorCode MatGetOwnershipRange      (ADMat mat, PetscInt *m, PetscInt *n);
PetscErrorCode MatGetSize                (ADMat mat, PetscInt *m, PetscInt *n);
PetscErrorCode MatGetType                (ADMat mat, MatType *type);
PetscErrorCode MatGetValue               (ADMat mat, PetscInt row, PetscInt col, Number* v);
PetscErrorCode MatGetValues              (ADMat mat, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], Number v[]);
PetscErrorCode MatMPIAIJSetPreallocation (ADMat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[]);
PetscErrorCode MatMult                   (ADMat mat, ADVec x, ADVec y);
PetscErrorCode MatNorm                   (ADMat x, NormType type, Number *val);
PetscErrorCode MatSeqAIJSetPreallocation (ADMat B, PetscInt nz, const PetscInt nnz[]);
PetscErrorCode MatSetFromOptions         (ADMat mat);
PetscErrorCode MatSetOption              (ADMat x, MatOption op, PetscBool flag);
PetscErrorCode MatSetSizes               (ADMat mat, PetscInt m, PetscInt n, PetscInt M, PetscInt N);
PetscErrorCode MatSetValue               (ADMat mat, PetscInt i, PetscInt j, Number v, InsertMode addv);
PetscErrorCode MatSetValues              (ADMat mat, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], const Number v[], InsertMode addv);
PetscErrorCode MatView                   (ADMat mat, PetscViewer viewer);
PetscErrorCode MatZeroEntries            (ADMat mat);

/*-------------------------------------------------------------------------------------------------
 * AD specific functions
 */

void ADMatCreateADData  (ADMat mat);
void ADMatCopyForReverse(ADMat mat, Mat* newm, ADMatData** newd);
void ADMatIsActive      (ADMat mat, bool* a);

/*-------------------------------------------------------------------------------------------------
 * Debug functions
 */

void ADMatDebugOutput(ADMat mat, std::string m, int id);
void ADMatDebugOutput(Mat mat, std::string m, int id);

AP_NAMESPACE_END
