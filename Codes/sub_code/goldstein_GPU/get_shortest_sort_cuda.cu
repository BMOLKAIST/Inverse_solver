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
#include "get_shortest_sort_mex.hpp"
#include "mex.h"

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
void dipole_compute_kernel_test(float * const pResidue, int32_t* const pPositive, float * const pOutArray, float * const pOut2Array,
	int const dim_out_1, int const dim_out_2, int const dim_out_3, int const dims_Positive, int const direction)
{
	
}

__global__
void set_to_minus_1(int32_t * const to_init, const int to_init_size)
{
	int id_1D = threadIdx.x + blockIdx.x * blockDim.x;

	if (id_1D < to_init_size) { // because can be change to negative if all use when removing dipole along another direction
		to_init[id_1D] = -1;
	}
}

#define GPU_COMPUTE // so the function in search universal ar addapted for gpu use
#include "search_universal.cpp"

int shortest_sort_KERNEL(int32_t const * const pResidue, int32_t const * const pLookup, int32_t const * const pLookup_z, int32_t const * const pSquareSize, int32_t * const pNearest_res, int residue_number_1D, int precompute_number, int32_t* const size_residue_3D, int lookup_size)
{

	int const size_3D_1 = size_residue_3D[0];
	int const size_3D_2 = size_residue_3D[1];
	int const size_3D_3 = size_residue_3D[2];

	int blockSize;
	int minGridSize;
	int gridSize;

	
	//find the nearest neighbour

	int arrayCount = residue_number_1D; // because itterates for each positive residue

	cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,(void*)shortest_compute_kernel,0,arrayCount);

	// Round up according to array size
	gridSize = (arrayCount + blockSize - 1) / blockSize;
	
		
	shortest_compute_kernel << <gridSize, blockSize >> > (pResidue, pLookup, pLookup_z, pSquareSize, pNearest_res, residue_number_1D, precompute_number, size_3D_1,  size_3D_2,  size_3D_3, lookup_size);
	
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());
	//cudaDeviceSynchronize();
	return 1;
}
