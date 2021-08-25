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
#include "create_sort_value_mex.hpp"
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
void sort_compute_kernel(int32_t const* const pResidue, int32_t const* const pSquareSize, int32_t* const pCaseID, int32_t* const pDim1ID, int32_t* const pDim2ID, int32_t const size_3D_1, int32_t const size_3D_2, int32_t const size_3D_3, int32_t const size_residue_1D)

{
	int id_1D = threadIdx.x + blockIdx.x * blockDim.x;

	if (id_1D < size_residue_1D ) { // because can be change to negative if all use when removing dipole along another direction

		int id_residue = pResidue[id_1D];//the position of the annalyzed residue 

		//return;

		//the position in 3D space
		int d12 = id_residue % ((size_3D_1)* (size_3D_2));
		int d3 = id_residue / ((size_3D_1)* (size_3D_2));
		int32_t d2 = d12 / (size_3D_1);
		int32_t d1 = d12 % (size_3D_1);

		if (d3 < size_3D_3) {

			pDim1ID[id_1D]=d1;
			pDim2ID[id_1D]=d2;

			int case_size = pSquareSize[d3];

			int case_side_1 = (int)ceilf(((float)size_3D_1) / ((float)case_size));
			int case_side_2 = (int)ceilf(((float)size_3D_2) / ((float)case_size));

			int id_case_1 = ((int)d1) / ((int)case_side_1);
			int id_case_2 = ((int)d2) / ((int)case_side_2);

			pCaseID[id_1D] = id_case_1 + id_case_2 * case_size;

		}
	}
}

int create_sort_KERNEL(int32_t const* const pResidue, int32_t const* const pSquareSize, int32_t* const pCaseID, int32_t* const pDim1ID, int32_t* const pDim2ID, int32_t* const size_residue_3D, int32_t size_residue_1D)
{

	//put the dimension in simpler variables
	int const size_3D_1 = size_residue_3D[0];
	int const size_3D_2 = size_residue_3D[1];
	int const size_3D_3 = size_residue_3D[2];

	//variables for launch configuration
	int blockSize;
	int minGridSize;
	int gridSize;

	int arrayCount = size_residue_1D; // because itterates for each positive residue

	cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,(void*)sort_compute_kernel,0,arrayCount);

	// Round up according to array size
	gridSize = (arrayCount + blockSize - 1) / blockSize;

	
	//direction = 0 -> top ; direction = 1 -> right ; direction = 2 -> bottom ; direction = 3 -> left ; 
	sort_compute_kernel << <gridSize, blockSize >> > (pResidue, pSquareSize, pCaseID, pDim1ID, pDim2ID, size_3D_1, size_3D_2, size_3D_3 , size_residue_1D);
	
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());
	//cudaDeviceSynchronize();
	return 1;
}
