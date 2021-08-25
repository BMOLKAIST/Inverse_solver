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
void dipole_compute_kernel(float * const pResidue, int32_t* const pPositive, float * const pOutArray, float * const pOut2Array,
	int const dim_out_1, int const dim_out_2, int const dim_out_3, int const dims_Positive,int const direction)
{
	int id_positive = threadIdx.x + blockIdx.x * blockDim.x;

	if (id_positive < dims_Positive && pPositive[id_positive] > 0) { // because can be change to negative if all use when removing dipole along another direction

		int id_residue = pPositive[id_positive] - 1;//the position of the annalyzed residue // not forget -1 because they are given as matlab

		//return;

		//the position in 3D space
		int d12 = id_residue % ((dim_out_1-1)* (dim_out_2-1));
		int d3 = id_residue / ((dim_out_1-1)* (dim_out_2-1));
		int d2 = d12 / (dim_out_1-1);
		int d1 = d12 % (dim_out_1-1);

		if (d3 < dim_out_3) {

			//check for dipole in the given direction


			int d1_pair = d1 - (direction == 0 ? 1 : 0) + (direction == 2 ? 1 : 0);
			int d2_pair = d2 - (direction == 1 ? 1 : 0) + (direction == 3 ? 1 : 0);

			if (d1_pair >= 0 && d2_pair >= 0 && d1_pair < (dim_out_1 - 1) && d2_pair < (dim_out_2 - 1) ) {

				int id_pair = d3 * (dim_out_1 - 1)*(dim_out_2 - 1) + (d2_pair) * (dim_out_1 - 1) + (d1_pair);// id of the other part of the dipole
				float pair_val = pResidue[id_pair];

				if ( pair_val < 0) {
					// pair detected
					float val_original = pResidue[id_residue];

					float val_to_remove = fminf(val_original, - pair_val);//not forget minus because the over value is negative

					val_original = roundf(val_original - val_to_remove);
					pair_val = roundf(pair_val + val_to_remove);

					pResidue[id_residue] = val_original;
					pResidue[id_pair] = pair_val;

					if (val_original == 0) {
						pPositive[id_positive] = -1; // so it is not recomputed 
					}

					if (direction == 1 || direction == 3) { // no link to the border so nothing cuts the first unwraping step for the second only change along dim 1 are taken into account

						int id_center_out1 = d1_pair + 1;// id of the center of the dipole in the output1 // the +1 is just an arbitrary choice
						int id_center_out2 = fminf(d2, d2_pair)+1;// id of the center of the dipole in the output1

						int id_pair = d3*(dim_out_1)*(dim_out_2) + id_center_out2*(dim_out_1) + id_center_out1;// id of the other part of the dipole

						pOutArray[id_pair] = (direction == 1 ? val_to_remove : - val_to_remove );

					}
				}
			}
		}
	}
}

int dipoles_KERNEL(float* const pResidue, int32_t* const pPositive, float * const pOutArray, float * const pOut2Array, int const * const dims_output, int const dims_Positive) 
{

	//put the dimension in simpler variables
	int const dim_out_1 = dims_output[0];
	int const dim_out_2 = dims_output[1];
	int const dim_out_3 = dims_output[2];

	//variables for launch configuration
	int blockSize;
	int minGridSize;
	int gridSize;

	int arrayCount = dims_Positive; // because itterates for each positive residue

	cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,(void*)dipole_compute_kernel,0,arrayCount);

	// Round up according to array size
	gridSize = (arrayCount + blockSize - 1) / blockSize;

	for (int direction = 0; direction < 4; direction++) {
		//direction = 0 -> top ; direction = 1 -> right ; direction = 2 -> bottom ; direction = 3 -> left ; 
		dipole_compute_kernel << <gridSize, blockSize >> > (pResidue, pPositive, pOutArray, pOut2Array,
			dim_out_1, dim_out_2, dim_out_3, dims_Positive, direction);
	}
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());
	//cudaDeviceSynchronize();
	return 1;
}
