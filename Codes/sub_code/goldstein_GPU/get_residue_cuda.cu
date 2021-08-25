/**
 * @file pctdemo_life_mex_shmem.cu
 * @brief Example of implementing a stencil operation on the GPU using shared memory.
 *
 * Copyright 2013 The MathWorks, Inc.
 */

#include <algorithm>
#include <math.h>
#include <cuda_runtime_api.h>
#include "get_residue_mex.hpp"
#include "mex.h"

 /**
  * Host function called by MEX gateway. Sets up and calls the device function
  * for each generation.
  */


#define CUDART_PI_F 3.141592654f

__device__
void complex_mult(const float & a_1, const float & a_2, const float & b_1, const float & b_2, float* c_1, float* c_2) {
	*c_1 = a_1 * b_1 - a_2 * b_2;
	*c_2 = a_1 * b_2 + a_2 * b_1;
}

__device__
float modulo_float(const float & a, const float & b) {
	return fmodf(fmodf(a,b) + b , b);
}

__global__
void compute_kernel(float const * const pPhase, float * const pOutArray,
	int const dim_out_1, int const dim_out_2, int const dim_out_3)
{

	int id_out = threadIdx.x + blockIdx.x * blockDim.x;
	int d12 = id_out % (dim_out_1* dim_out_2);
	int d3 = id_out / (dim_out_1* dim_out_2);
	int d2 = d12 / dim_out_1;
	int d1 = d12 % dim_out_1;

	if (d3 < dim_out_3) {

		int id_in_11 = d3 * (dim_out_1 + 1)*(dim_out_2 + 1) + d2 * (dim_out_1 + 1) + d1;
		int id_in_21 = d3 * (dim_out_1 + 1)*(dim_out_2 + 1) + d2 * (dim_out_1 + 1) + d1 + 1;
		int id_in_12 = d3 * (dim_out_1 + 1)*(dim_out_2 + 1) + (d2 + 1) * (dim_out_1 + 1) + d1;
		int id_in_22 = d3 * (dim_out_1 + 1)*(dim_out_2 + 1) + (d2 + 1) * (dim_out_1 + 1) + d1 + 1;

		float val_11 = pPhase[id_in_11];
		float val_21 = pPhase[id_in_21];
		float val_12 = pPhase[id_in_12];
		float val_22 = pPhase[id_in_22];

		float d1_1 = modulo_float(val_11 - val_12 + CUDART_PI_F, 2.0f*CUDART_PI_F);// -CUDART_PI_F;
		float d1_2 = modulo_float(val_21 - val_22 + CUDART_PI_F, 2.0f*CUDART_PI_F);// -CUDART_PI_F;
		float d2_1 = modulo_float(val_11 - val_21 + CUDART_PI_F, 2.0f*CUDART_PI_F);// -CUDART_PI_F;
		float d2_2 = modulo_float(val_12 - val_22 + CUDART_PI_F, 2.0f*CUDART_PI_F);// -CUDART_PI_F;

		float residue = roundf((d1_1 + d2_2 - d1_2 - d2_1) / (2.0f * CUDART_PI_F));

		pOutArray[id_out] = residue;

	}
}

int residue_KERNEL(float const * const pPhase, float * const pOutArray, int const * const dims_output)
{
	//put the dimension in simpler variables
	int const dim_out_1 = dims_output[0];
	int const dim_out_2 = dims_output[1];
	int const dim_out_3 = dims_output[2];

	//variables for launch configuration
	int blockSize;
	int minGridSize;
	int gridSize;

	int arrayCount = dim_out_1 * dim_out_2 * dim_out_3;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,(void*)compute_kernel,0,arrayCount);


	// Round up according to array size
	gridSize = (arrayCount + blockSize - 1) / blockSize;


	compute_kernel << <gridSize, blockSize >> > ( pPhase, pOutArray,
		 dim_out_1, dim_out_2, dim_out_3);
	//cudaDeviceSynchronize();
	return 1;
}
