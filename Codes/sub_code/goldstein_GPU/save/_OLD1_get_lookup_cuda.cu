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
#include "get_lookup_mex.hpp"
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
void set_to_minus_1(int32_t * const pLookupTable, const int size_lookup_table)
{
	int id_1D = threadIdx.x + blockIdx.x * blockDim.x;

	if (id_1D < size_lookup_table) { // because can be change to negative if all use when removing dipole along another direction
		pLookupTable[id_1D] = -1;
	}
}
__global__
void lookup_compute_kernel(int32_t * const pLookupTable, int32_t const * const plookup_z_start, int32_t const * const pResidue,const int dim_1_residue)
{
	int id_1D = threadIdx.x + blockIdx.x * blockDim.x;

	if (id_1D < dim_1_residue) { // because can be change to negative if all use when removing dipole along another direction

		const int case_ID_colomn = 2;
		const int z_ID_colomn = 5;

		const int row_num = 6;
		//current residue data
		int case_current_residue = pResidue[case_ID_colomn + row_num*id_1D];
		int z_current_residue    = pResidue[z_ID_colomn + row_num*id_1D];
		//last residue data
		int case_last_residue = -1;
		int z_last_residue = -1;
		if (id_1D > 0) {
			case_last_residue = pResidue[case_ID_colomn + row_num*(id_1D - 1)];
			z_last_residue    = pResidue[z_ID_colomn + row_num*(id_1D - 1)];
		}

		//check if need to update the lookup table

		if (case_current_residue != case_last_residue || z_current_residue != z_last_residue) {
			int lookup_ID = plookup_z_start[z_current_residue] + case_current_residue;//the id in the lookup table to update 
			pLookupTable[lookup_ID] = id_1D;// update it with the 1D index to the given element
		}
	}
}

int lookup_KERNEL(int32_t * const pLookupTable, int32_t const * const plookup_z_start, int32_t const * const pResidue, int dim_1_residue, int size_lookup_table)
{

	int blockSize;
	int minGridSize;
	int gridSize;

	int arrayCount = size_lookup_table; // because itterates for each positive residue

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)set_to_minus_1, 0, arrayCount);

	// Round up according to array size
	gridSize = (arrayCount + blockSize - 1) / blockSize;


	set_to_minus_1 << <gridSize, blockSize >> > (pLookupTable, size_lookup_table);

	//put the dimension in simpler variables
	//int const size_3D_1 = size_residue_3D[0];
	//int const size_3D_2 = size_residue_3D[1];
	//int const size_3D_3 = size_residue_3D[2];

	//variables for launch configuration
	//int blockSize;
	//int minGridSize;
	//int gridSize;

	 arrayCount = dim_1_residue; // because itterates for each positive residue

	cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,(void*)lookup_compute_kernel,0,arrayCount);

	// Round up according to array size
	gridSize = (arrayCount + blockSize - 1) / blockSize;


	lookup_compute_kernel << <gridSize, blockSize >> > ( pLookupTable, plookup_z_start, pResidue, dim_1_residue);
	
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());
	//cudaDeviceSynchronize();
	return 1;
}
