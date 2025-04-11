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

#include <adjoint_petsc/util/base.hpp>
#include <adjoint_petsc/util/codi_def.hpp>
#include <adjoint_petsc/mat.h>

AP_NAMESPACE_START

enum class ADMatType {
  NONE = 0,
  ADMatAIJ,    // Not used it is directly mapped to ADMatSeqMAIJ or ADMatMPIMAIJ
  ADMatSeqAIJ,
  ADMatMPIAIJ
};


struct ADMatSeqAIJData : public ADMatData {
  static int constexpr TYPE = (int)ADMatType::ADMatSeqAIJ;

  std::vector<Identifier> index;

  ADMatSeqAIJData(ADMatSeqAIJData const&) = default;
  explicit ADMatSeqAIJData(PetscInt size);

  ADMatSeqAIJData* clone() override;

  static ADMatSeqAIJData* cast(ADMatData* d);
};

struct ADMatMPIAIJData : public ADMatData {
  static int constexpr TYPE = (int)ADMatType::ADMatMPIAIJ;
  ADMatSeqAIJData index_d;
  ADMatSeqAIJData index_o;

  ADMatMPIAIJData(ADMatMPIAIJData const&) = default;
  ADMatMPIAIJData(PetscInt diag_size, PetscInt off_diag_size);

  ADMatMPIAIJData* clone() override;

  static ADMatMPIAIJData* cast(ADMatData* d);
};

ADMatType ADMatDataPTypeToEnum(MatType ptype);

ADMatType ADMatGetADType(ADMat mat);
ADMatType ADMatGetADType(Mat mat);

AP_NAMESPACE_END
