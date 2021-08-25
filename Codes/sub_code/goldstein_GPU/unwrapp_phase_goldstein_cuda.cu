/**
 * @file pctdemo_life_mex_shmem.cu
 * @brief Example of implementing a stencil operation on the GPU using shared memory.
 *
 * Copyright 2013 The MathWorks, Inc.
 */
#include <stdint.h>
#include <algorithm>
#include <math.h>
#include <cuda_runtime_api.h>
#include "unwrapp_phase_goldstein_mex.hpp"
#include "mex.h"
#include <array>

 /**
  * Host function called by MEX gateway. Sets up and calls the device function
  * for each generation.
  */


#define CUDART_PI_F 3.141592654f

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		mexPrintf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

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
void set_to_minus_1(int32_t * const pLookupTable, const int size_lookup_table)
{
	int id_1D = threadIdx.x + blockIdx.x * blockDim.x;

	if (id_1D < size_lookup_table) { // because can be change to negative if all use when removing dipole along another direction
		pLookupTable[id_1D] = -1;
	}
}

__device__ int sign(int x)

{
	return x > 0 ? 1 : -1;
}

__global__
void unwrapp_kernel_1(float * const pPhase, int32_t const * const pStep1, int32_t const * const pStep2,const int size_Phase_1, const int size_Phase_2, const int size_Phase_3)
{
	int id_1D = threadIdx.x + blockIdx.x * blockDim.x;

	if (id_1D < size_Phase_3) { // because can be change to negative if all use when removing dipole along another direction
		
		float last_phase = pPhase[id_1D * size_Phase_2 * size_Phase_1];//the first element which is not unwrapped
		float current_phase = 0;
		float phase_difference = 0;

		for (int i = 1; i < size_Phase_2; i++) {//no need to unwrapp the first since it has no preceeding values
			current_phase = pPhase[id_1D * size_Phase_2 * size_Phase_1 + i * size_Phase_1];

			phase_difference = modulo_float(current_phase - last_phase + CUDART_PI_F, 2.0f*CUDART_PI_F) - CUDART_PI_F;

			current_phase = last_phase + phase_difference - ((float)pStep2[id_1D * size_Phase_2 + i])*2.0f*CUDART_PI_F;

			pPhase[id_1D * size_Phase_2 * size_Phase_1 + i * size_Phase_1] = current_phase;
			last_phase = current_phase;
		}

	}
}

__global__
void unwrapp_kernel_2(float * const pPhase, int32_t const * const pStep1, int32_t const * const pStep2, const int size_Phase_1, const int size_Phase_2, const int size_Phase_3)
{
	int id_1D = threadIdx.x + blockIdx.x * blockDim.x;

	if (id_1D < size_Phase_2*size_Phase_3) { // because can be change to negative if all use when removing dipole along another direction

		int id_2D_3 = id_1D / size_Phase_2;
		int id_2D_2 = id_1D % size_Phase_2;

		float last_phase = pPhase[id_2D_3 * size_Phase_2 * size_Phase_1 + id_2D_2 * size_Phase_1];//the first element which is not unwrapped
		float current_phase = 0;
		float phase_difference = 0;

		for (int i = 1; i < size_Phase_1; i++) {//no need to unwrapp the first since it has no preceeding values

			current_phase = pPhase[id_2D_3 * size_Phase_2 * size_Phase_1 + id_2D_2 * size_Phase_1 + i];

			phase_difference = modulo_float(current_phase - last_phase + CUDART_PI_F, 2.0f*CUDART_PI_F) - CUDART_PI_F;

			current_phase = last_phase + phase_difference - ((float)pStep1[id_2D_3 * size_Phase_2 * size_Phase_1 + id_2D_2 * size_Phase_1 + i])*2.0f*CUDART_PI_F;

			pPhase[id_2D_3 * size_Phase_2 * size_Phase_1 + id_2D_2 * size_Phase_1 + i] = current_phase;
			last_phase = current_phase;

		}

	}
}

int unwrapp_KERNEL(float * const pPhase, int32_t const * const pStep1, int32_t const * const pStep2, std::array<int, 3> size_Phase)
{

	int size_Phase_1 = size_Phase[0];
	int size_Phase_2 = size_Phase[1];
	int size_Phase_3 = size_Phase[2];

	//mexPrintf("size %d : %d : %d \n", size_Step1_1, size_Step1_2, size_Step1_3);

	int blockSize;
	int minGridSize;
	int gridSize;


	//UNWRAPP 1D

	int arrayCount = size_Phase_3; // because itterates for each positive residue

	cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,(void*)unwrapp_kernel_1,0,arrayCount);

	gridSize = (arrayCount + blockSize - 1) / blockSize;// Round up according to array size

	unwrapp_kernel_1 << <gridSize, blockSize >> > (pPhase, pStep1, pStep2, size_Phase_1, size_Phase_2, size_Phase_3);


	//UNWARP 2D


	arrayCount = size_Phase_2* size_Phase_3; // because itterates for each positive residue

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)unwrapp_kernel_2, 0, arrayCount);

	gridSize = (arrayCount + blockSize - 1) / blockSize;// Round up according to array size

	unwrapp_kernel_2 << <gridSize, blockSize >> > (pPhase, pStep1, pStep2, size_Phase_1, size_Phase_2, size_Phase_3);
	
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());
	//cudaDeviceSynchronize();
	return 1;
}
