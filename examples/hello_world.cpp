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

#include <petscsys.h>

#include <adjoint_petsc/ksp.h>
#include <adjoint_petsc/mat.h>
#include <adjoint_petsc/vec.h>

using Number  = adjoint_petsc::Number;
using Tape  = adjoint_petsc::Tape;
using ADKSP = adjoint_petsc::ADKSP;
using ADMat = adjoint_petsc::ADMat;
using ADVec = adjoint_petsc::ADVec;

  static int constexpr ENTRIES_PER_RANK        = 4;
  static int constexpr MATRIX_NONZERSO_PER_ROW = 2;

int main(int argc, char** argv) {
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));

  int mpi_rank;
  int mpi_size;

  MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);

  ADVec x;
  ADVec rhs;
  ADMat A;
  ADKSP ksp;

  // Create x
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, ENTRIES_PER_RANK, ENTRIES_PER_RANK * mpi_size));
  PetscCall(VecSetFromOptions(x));

  // Create rhs
  PetscCall(VecDuplicate(x, &rhs));

  // Create A
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, ENTRIES_PER_RANK, ENTRIES_PER_RANK, ENTRIES_PER_RANK* mpi_size, ENTRIES_PER_RANK * mpi_size, MATRIX_NONZERSO_PER_ROW, NULL, MATRIX_NONZERSO_PER_ROW, NULL, &A));

  // Create KSP
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetFromOptions(ksp));


  // Init AD
  Tape& tape = Number::getTape();
  Number inputRhs = 1.0;
  Number inputADiag = 2.0;
  Number inputAOffDiag = 1.0;
  tape.setActive();
  tape.registerInput(inputRhs);
  tape.registerInput(inputADiag);
  tape.registerInput(inputAOffDiag);

  // Init rhs
  if(mpi_size == mpi_rank + 1) {
    adjoint_petsc::WrapperArray values = {};
    PetscCall(VecGetArray(rhs, &values));
    values[ENTRIES_PER_RANK - 1] = inputRhs;
    PetscCall(VecRestoreArray(rhs, &values));
  }

  // Init A
  int low, high;
  PetscCall(MatGetOwnershipRange(A, &low, &high));

  int matSize = ENTRIES_PER_RANK * mpi_size;

  for(int i = 0; i < ENTRIES_PER_RANK; i += 1) {
    PetscCall(  MatSetValue(A, i + low, i + low,     inputADiag,    INSERT_VALUES));
    if(i + 1 + low < matSize) { // Skip off diag entry on last row.
      PetscCall(MatSetValue(A, i + low, i + 1 + low, inputAOffDiag, INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Init ksp
  PetscCall(KSPSetOperators(ksp, A, A));

  // Solve
  PetscCall(KSPSolve(ksp, rhs, x));
  Number norm;
  PetscCall(VecNorm(x, NORM_2, &norm));

  // Init reverse AD
  tape.registerOutput(norm);
  tape.setPassive();

  if(0 == mpi_rank) { // Only seed norm once.
    norm.setGradient(1.0);
  }
  tape.evaluate();

  // Output
  if(0 == mpi_rank) { std::cout << "A = " << std::endl; }   MPI_Barrier(PETSC_COMM_WORLD);
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  if(0 == mpi_rank) { std::cout << "\n\nrhs = " << std::endl; } MPI_Barrier(PETSC_COMM_WORLD);
  PetscCall(VecView(rhs, PETSC_VIEWER_STDOUT_WORLD));
  if(0 == mpi_rank) { std::cout << "\n\nx = " << std::endl; }   MPI_Barrier(PETSC_COMM_WORLD);
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  // AD Output
  for(int i = 0; i < mpi_size; i += 1) {
    if(i == mpi_rank) {
      std::cout << "\n\nRank " << i << std::endl;
      std::cout << "inputRhs_b      = " << inputRhs.getGradient() << std::endl;
      std::cout << "inputADiag_b    = " << inputADiag.getGradient() << std::endl;
      std::cout << "inputAOffDiag_b = " << inputAOffDiag.getGradient() << std::endl;
    }
    MPI_Barrier(PETSC_COMM_WORLD);
  }

  tape.reset();

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&x));

  PetscCall(PetscFinalize());

  return 0;
}
