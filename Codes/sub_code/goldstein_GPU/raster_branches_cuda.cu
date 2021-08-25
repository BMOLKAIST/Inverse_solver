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
#include "raster_branches_mex.hpp"
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

__device__ int sign(int x)

{
	return x > 0 ? 1 : -1;
}

__global__
void raster_compute_kernel(int32_t const * const pBranches, int32_t * const pStep1, int32_t * const pStep2, const int size_Branches, const int size_Step1_1, const int size_Step1_2, const int size_Step1_3)
{
	int id_1D = threadIdx.x + blockIdx.x * blockDim.x;

	if (id_1D < size_Branches) { // because can be change to negative if all use when removing dipole along another direction
		//get data

		const int column_num = 6;
		int point_1_x = pBranches[id_1D*column_num];
		int point_1_y = pBranches[id_1D*column_num + 1];
		int point_2_x = pBranches[id_1D*column_num + 2];
		int point_2_y = pBranches[id_1D*column_num + 3];
		int point_z   = pBranches[id_1D*column_num + 4];
		int jump      = pBranches[id_1D*column_num + 5];

		if (point_1_x == -1) {//link point 2 to a border
			//first compute the real coordinates of point 1
			bool x_side = point_2_x < size_Step1_1 - point_2_x;
			bool y_side = point_2_y < size_Step1_2 - point_2_y;

			int min_distance_x = (x_side ? point_2_x : size_Step1_1 - point_2_x);
			int min_distance_y = (y_side ? point_2_y : size_Step1_2 - point_2_y);

			point_1_x = ( min_distance_x > min_distance_y ? point_2_x : (x_side ? -1 : size_Step1_1-1));
			point_1_y = ( min_distance_x < min_distance_y ? point_2_y : (y_side ? -1 : size_Step1_2-1));
			// verify if need to update pStep2
			if (point_1_x == -1) {
				// need to update pStep2
				atomicAdd(&pStep2[size_Step1_2*point_z + point_2_y + 1],jump);//plus one because the coordinate of the point and pStep are shifted of 0.5
				return;
			}
			if (point_1_x == size_Step1_1) {
				//vertical and not touching bottom so nothing to draw
				return;
			}
		}

		// scan all the orisontal positions of the branche

		if (point_1_y != point_2_y) {
			if (point_2_y > point_1_y) {
				int tmp;
				tmp = point_1_y;
				point_1_y = point_2_y;
				point_2_y = tmp;
				tmp = point_1_x;
				point_1_x = point_2_x;
				point_2_x = tmp;
				jump = -jump;
			}
			int steps =  point_1_y - point_2_y;//positive number
			for (int i = 0; i < steps; i++) {
				int coor_y = point_2_y +  i + 1;//plus one because the coordinate of the point and pStep are shifted of 0.5
				int coor_x = point_2_x + (int)(((float)i/(float)steps)*((float)(point_1_x-point_2_x))) + 1;//plus one because the coordinate of the point and pStep are shifted of 0.5

				atomicAdd(&pStep1[size_Step1_1*size_Step1_2*point_z + size_Step1_1*coor_y + coor_x], jump );
			}

		}

		// raster the branch
		// first check for border
	}
}

int raster_KERNEL(int32_t const * const pBranches, int32_t * const pStep1, int32_t * const pStep2, const int size_Branches, const std::array<int, 3> size_Step1)
{

	int size_Step1_1 = size_Step1[0];
	int size_Step1_2 = size_Step1[1];
	int size_Step1_3 = size_Step1[2];

	//mexPrintf("size %d : %d : %d \n", size_Step1_1, size_Step1_2, size_Step1_3);

	int blockSize;
	int minGridSize;
	int gridSize;

	int arrayCount = size_Branches; // because itterates for each positive residue


	cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,(void*)raster_compute_kernel,0,arrayCount);

	// Round up according to array size
	gridSize = (arrayCount + blockSize - 1) / blockSize;


	raster_compute_kernel << <gridSize, blockSize >> > (pBranches, pStep1, pStep2, size_Branches, size_Step1_1, size_Step1_2, size_Step1_3);
	
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());
	//cudaDeviceSynchronize();
	return 1;
}
