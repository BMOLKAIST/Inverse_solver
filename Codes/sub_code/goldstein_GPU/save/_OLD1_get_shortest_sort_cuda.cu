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

/*
__device__
int binarySearch_array_2_size(int arr[], int x, int low, int high)
{
	int mid;
	while (low < high) {
		mid = (high + low) / 2;
		if (arr[2 * mid + 1] == x) {
			break;
		}
		else if (arr[2 * mid + 1] > x) {
			high = mid - 1;
		}
		else {
			low = mid + 1;
		}
	}
	mid = (high + low) / 2;
	if (x <= arr[2 * mid + 1])
		return mid;
	else
		return mid + 1;
}
__device__
void search_in_case(int32_t const * const pLookup,int32_t const * const pResidue, int * const current_shortest, const int square_size, const int box_1, const int box_2,const int start_lookup, const int residue_number_1D, const int lookup_size, const int x_current_residue, const int y_current_residue, const int precompute_number , const int last_dist,const int id_1D) {
	
	if (box_1 >= square_size || box_2 >= square_size) {
		//out of the region
		return;
	}
	int case_look_id = start_lookup + box_1 + box_2 * square_size;
	int start_look = pLookup[case_look_id];
	if (start_look == -1) {
		// this box is empty
		return;
	}
	int end_look = residue_number_1D;
	int next_full = 1;
	int val_look;
	while (case_look_id + next_full < lookup_size && end_look == residue_number_1D) {
		val_look = pLookup[case_look_id + next_full];
		if (val_look >= 0) {//because if next box is empty reading would be erronous
			end_look = val_look;
			break;
		}
		next_full++;
	}
	//if(case_look_id + 1 < lookup_size)end_look = pLookup[case_look_id + 1];

	const int x_ID_colomn = 3;
	const int y_ID_colomn = 4;
	const int row_num = 6;

	int dim1, dim2,distance,insertion;
	int exec = 0;
	for (int i = start_look; i < end_look; i++) {
		dim1 = pResidue[x_ID_colomn + row_num * i];
		dim2 = pResidue[y_ID_colomn + row_num * i];
		exec++;
		
		int x_coor = abs(x_current_residue - dim1);
		int y_coor = abs(y_current_residue - dim2);

		//distance = max(x_coor,y_coor);//manhattan distance
		distance = sqrtf(x_coor*x_coor + y_coor*y_coor);//euclidian distance

		int current_last_dist = current_shortest[precompute_number * 2 - 1];
		if (distance < current_last_dist && i != id_1D) {//also check that it is not itself
			insertion = binarySearch_array_2_size(current_shortest, distance, 0, precompute_number - 1);
			// if found an insertion point
			if (insertion < precompute_number) {
				for (int k = precompute_number - 1; k > insertion ; k--) {
					current_shortest[2 * k] = current_shortest[2 * (k - 1)];
					current_shortest[2 * k + 1] = current_shortest[2 * (k - 1) + 1];
				}
				current_shortest[2 * insertion] = i;
				current_shortest[2 * insertion + 1] = distance;
			}
		}
	}

}

__global__
void shortest_compute_kernel(int32_t const * const pResidue, int32_t const * const pLookup, int32_t const * const pLookup_z, int32_t const * const pSquareSize, int32_t * const pNearest_res, const int residue_number_1D, const int precompute_number, int32_t const size_3D_1, int32_t const size_3D_2, int32_t const size_3D_3, int lookup_size)
{
	int id_1D = threadIdx.x + blockIdx.x * blockDim.x;

	if (id_1D < residue_number_1D) { // because can be change to negative if all use when removing dipole along another direction

		const int case_ID_colomn = 2;
		const int x_ID_colomn = 3;
		const int y_ID_colomn = 4;
		const int z_ID_colomn = 5;

		const int row_num = 6;
		//current residue data
		int case_current_residue = pResidue[case_ID_colomn + row_num*id_1D];
		int x_current_residue = pResidue[x_ID_colomn + row_num*id_1D];
		int y_current_residue = pResidue[y_ID_colomn + row_num*id_1D];
		int z_current_residue = pResidue[z_ID_colomn + row_num*id_1D];
		// starting point of the current z section in the lookup
		const int start_lookup = pLookup_z[z_current_residue];
		//square size at the current z depth
		int square_size = pSquareSize[z_current_residue];
		// get the number of case per side at the current depth
		int case_side_1 = (int)ceilf(((float)size_3D_1) / ((float)square_size));
		int case_side_2 = (int)ceilf(((float)size_3D_2) / ((float)square_size));
		//bicoordinate of case in which the residue is
		int case_current_residue_1 = case_current_residue % square_size;
		int case_current_residue_2 = case_current_residue / square_size;
		int max_border_distance = max(max(square_size - case_current_residue_1, square_size - case_current_residue_2),max(case_current_residue_1, case_current_residue_2));
		//search in adjacent cases
		int * current_shortest = new int[2*precompute_number];
		
		//finding algorithm
		int box_search_distance = 0;
		int distance_max = 200000;// just a big number 
		// the maximum distance is the distance to border
		distance_max = min(min(
			x_current_residue, y_current_residue
			), min(
			size_3D_1-x_current_residue, size_3D_2-y_current_residue
			));
		for (int ii = 0; ii < precompute_number ; ++ii) {
			current_shortest[ii*2]   = -1;
			current_shortest[ii*2+1] = distance_max;// initialise to -1 --> not found
		}

		int last_dist = distance_max;
		int box_1[4] = { -1,-1,-1,-1 };// the coordinate of the search box along dimenssion 1
		int box_2[4] = { -1,-1,-1,-1 };// the coordinate of the search box along dimenssion 1
		int min_dist_in_search_box[4] = { distance_max,distance_max,distance_max,distance_max };//the min distance in the searched box

		int executed_searches = 0;

		while (box_search_distance < max_border_distance && box_search_distance < square_size) {
			//search in the area for shorter matches
			if (box_search_distance == 0) {
				//search the center
				box_1[0] = case_current_residue_1;
				box_2[0] = case_current_residue_2;
				search_in_case(pLookup, pResidue, current_shortest, square_size, box_1[0], box_2[0], start_lookup, residue_number_1D, lookup_size, x_current_residue, y_current_residue, precompute_number, last_dist, id_1D);
				executed_searches++;
			}
			else {
				for (int ii = 1-box_search_distance; ii <= box_search_distance; ++ii) {
					//top part
					box_1[0] = case_current_residue_1-box_search_distance;
					box_2[0] = case_current_residue_2-ii;
					min_dist_in_search_box[0] = max(abs(
						(box_1[0]+1)*case_side_1- x_current_residue
						),abs(
						((box_2[0] + (ii > 0 ? 1 : 0))*case_side_2 - y_current_residue)*(ii == 0 ? 0 : 1)
						));
					
					//right part
					box_1[1] = case_current_residue_1 + ii;
					box_2[1] = case_current_residue_2 - box_search_distance;
					min_dist_in_search_box[1] = max(abs(
						((box_1[1] + (ii > 0 ? 0 : 1))*case_side_1 - x_current_residue)*(ii == 0 ? 0 : 1)
						), abs(
						(box_2[1]+1)*case_side_2 - y_current_residue
						));
					
					//left part
					box_1[2] = case_current_residue_1 - ii;
					box_2[2] = case_current_residue_2 + box_search_distance;
					min_dist_in_search_box[2] = max(abs(
						((box_1[2] + (ii > 0 ? 1 : 0))*case_side_1 - x_current_residue)*(ii == 0 ? 0 : 1)
						), abs(
						(box_2[2])*case_side_2 - y_current_residue
						));
					
					//bottom part
					box_1[3] = case_current_residue_1 + box_search_distance;
					box_2[3] = case_current_residue_2 + ii;
					min_dist_in_search_box[3] = max(abs(
						(box_1[3])*case_side_1 - x_current_residue
						), abs(
						((box_2[3] + (ii > 0 ? 0 : 1))*case_side_2 - y_current_residue)*(ii == 0 ? 0 : 1)
						));
					
					bool to_eval[4] = { min_dist_in_search_box[0] < last_dist ,min_dist_in_search_box[1] < last_dist ,min_dist_in_search_box[2] < last_dist ,min_dist_in_search_box[3] < last_dist };
					int eval_num = 0;
					while (eval_num < 4) {//this strange loop is to avoid branche which can reduce performances in warps
						while (!to_eval[eval_num] && eval_num < 4) { eval_num++; }
						if (eval_num < 4) {

							search_in_case(pLookup, pResidue, current_shortest, square_size, box_1[eval_num], box_2[eval_num], start_lookup, residue_number_1D, lookup_size, x_current_residue, y_current_residue, precompute_number, last_dist, id_1D);
							executed_searches++;

						}
						eval_num++;
					}
					last_dist = current_shortest[precompute_number * 2 - 1];//update the farvest element in the list
				}
			}
			//if all the shortest ar found and next iteration wont give shorter then exit
			last_dist = current_shortest[precompute_number * 2 - 1];//update the farvest element in the list
			if (box_search_distance * case_side_1 >= last_dist && box_search_distance * case_side_2 >= last_dist) {
				break;
			}
			// increase the search radius to fing more
			box_search_distance++;
		}
		//udate the result to the mattrix
		for (int ii = 0; ii < precompute_number * 2; ++ii) pNearest_res[2 * precompute_number*id_1D + ii] =  current_shortest[ii];// return result
		delete[] current_shortest;
	}
}
*/
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
