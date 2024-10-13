//HEAD_DSPH
/*
 <DUALSPHYSICS>  Copyright (c) 2020 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/). 

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

 You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file JSphGpuSimple_ker.h \brief Declares functions and CUDA kernels for the Particle Interaction and System Update.

#ifndef _JSphGpuSimple_ker_
#define _JSphGpuSimple_ker_

#include "DualSphDef.h"
#include <cuda_runtime_api.h>

/// Implements a set of functions and CUDA kernels for the particle interaction and system update.
namespace cusphs{



//-Kernels to prepare data before Interaction_Forces().
//-------------------------------------------------------
void UpdatePosCell(unsigned np,tdouble3 posmin,float poscellsize
  ,const double2 *posxy,const double *posz,float4 *poscell,cudaStream_t stm);
void InitAceGravity(unsigned np,unsigned npb,tfloat3 gravity,float3 *ace,cudaStream_t stm);


//-Kernels to run after Interaction_Forces().
//---------------------------------------------
void Resety(unsigned n,unsigned ini,float3 *v,cudaStream_t stm);


//-Kernels for ComputeStep (vel & rhop).
//----------------------------------------
void ComputeStepVerlet(bool floating,bool shift,bool inout,unsigned np,unsigned npb
  ,const float4 *velrhop1,const float4 *velrhop2
  ,const float *ar,const float3 *ace,const float4 *shiftposfs,const float3 *indirvel
  ,double dt,double dt2,float rhopzero,float rhopoutmin,float rhopoutmax,tfloat3 gravity
  ,typecode *code,double2 *movxy,double *movz,float4 *velrhopnew,cudaStream_t stm);
void ComputeStepSymplecticPre(bool floating,bool shift,bool inout,bool hpdc,unsigned np,unsigned npb
  ,const float4 *velrhoppre,const float *ar,const float3 *ace,const float4 *shiftposfs
  ,const float3 *indirvel,double dtm,float rhopzero,float rhopoutmin,float rhopoutmax,tfloat3 gravity
  ,typecode *code,double2 *movxy,double *movz,float4 *velrhop, const float* psipre, float* psi, const float* apsi, cudaStream_t stm);
void ComputeStepSymplecticCor(bool floating,bool shift,bool inout,bool vem,bool vcs,bool hpdc,unsigned np,unsigned npb
  ,const float4 *velrhoppre,const float *ar,const float3 *ace,const float4 *shiftposfs, const float3 *acevem, const float vemthreshold, const float3 *shiftvcs
  ,const float3 *indirvel,double dtm,double dt,float rhopzero,float rhopoutmin,float rhopoutmax,tfloat3 gravity
  ,typecode *code,double2 *movxy,double *movz,float4 *velrhop, const float* psipre, float* psi, const float* apsi, cudaStream_t stm);

void ComputeStepSymplecticPreOSDP(bool floating, bool shift, bool inout, unsigned np, unsigned npb
	, const float4* velrhoppre, const double2* posxypre, const double* poszpre, const float* ar, const float3* ace, const float4* shiftposfs
	, const float3* indirvel, double dtm, float rhopzero, float rhopoutmin, float rhopoutmax, tfloat3 gravity
	, typecode* code, double2* movxy, double* movz, float4* velrhop, cudaStream_t stm);
void ComputeStepSymplecticCorOSDP(bool floating, bool shift, bool inout, bool vcs, unsigned np, unsigned npb
	, const float4* velrhoppre, const double2 *posxy,const double *posz,const float* ar, const float3* ace, const float4* shiftposfs, const float3* shiftvcs
	, const float3* indirvel, double dtm, double dt, float rhopzero, float rhopoutmin, float rhopoutmax, tfloat3 gravity
	, typecode* code, double2* movxy, double* movz, float4* velrhop, cudaStream_t stm);



void ComputeStepSymplecticPreOSDP2(bool floating, bool shift, bool inout, unsigned np, unsigned npb
	, const float4* velrhoppre, const double2* posxypre, const double* poszpre, const float* ar, const float3* ace, const float4* shiftposfs
	, const float3* indirvel, double dtm, float rhopzero, float rhopoutmin, float rhopoutmax, tfloat3 gravity
	, typecode* code, double2* movxy, double* movz, float4* velrhop, const float* psipre, float* psi, const float* apsi, cudaStream_t stm);
void ComputeStepSymplecticCorOSDP2(bool floating, bool shift, bool inout, bool vem, bool vcs, unsigned np, unsigned npb
	, const float4* velrhoppre, const double2* posxy, const double* posz, const float* ar, const float3* ace, const float4* shiftposfs, const float3* acevem, const float3* shiftvcs
	, const float3* indirvel, double dtm, double dt, float rhopzero, float rhopoutmin, float rhopoutmax, tfloat3 gravity
	, typecode* code, double2* movxy, double* movz, float4* velrhop, const float* psipre, float* psi, const float* apsi, cudaStream_t stm);
}


#endif


