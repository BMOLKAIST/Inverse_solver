/**
 * @file pctdemo_life_mex_texture.cpp
 * @brief MEX gateway for a stencil operation.
 * Copyright 2013 The MathWorks, Inc.
 *
 */

#include <stdint.h>
#include "tmwtypes.h"
#include "mex.h"
#include <stdint.h>
#include <math.h>
#include "get_branches_goldstein_compute.hpp"
#include "get_branches_goldstein_struct.hpp"
#include <vector>
#include <array>
#include <chrono>
#include <thread>

 /**
  * MEX gateway
  */
mwSize get_1D_array_size(const mwSize dim_size, const mwSize* dims) {
	if (dim_size == 1) {
		return dims[0];
	}
	else if (dim_size == 2) {
		if (dims[0] == 1) {
			return dims[1];
		}
		else if (dims[1] == 1) {
			return dims[0];
		}
		else {
			return 0;
		}
	}
	else {
		return 0;
	}
}

void mexFunction(int  nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "lapjv:InvalidInput";
	char const * const errMsg = "Provide all 6 input and as int32 to MEX file.";
	char const * const errMsg2 = "Size error";
	char const * const errMsg3 = "Too many output.";


	if (nrhs != 6) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}

	// retrieve the inputs

	mxArray const * const residues = (prhs[0]);
	if (!mxIsInt32(residues)) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}
	int32_t const * const presidues = static_cast<int32_t const *>(mxGetData(residues));

	mxArray const * const shortest_list = (prhs[1]);
	if (!mxIsInt32(shortest_list)) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}
	int32_t const * const pshortest_list = static_cast<int32_t const *>(mxGetData(shortest_list));

	mxArray const * const square_size = (prhs[2]);
	if (!mxIsInt32(square_size)) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}
	int32_t const * const psquare_size = static_cast<int32_t const *>(mxGetData(square_size));

	mxArray const * const lookup_table = (prhs[3]);
	if (!mxIsInt32(lookup_table)) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}
	int32_t const * const plookup_table = static_cast<int32_t const *>(mxGetData(lookup_table));

	mxArray const * const lookup_z_start = (prhs[4]);
	if (!mxIsInt32(lookup_z_start)) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}
	int32_t const * const plookup_z_start = static_cast<int32_t const *>(mxGetData(lookup_z_start));

	mxArray * const Sizes_3D = mxDuplicateArray(prhs[5]);
	if (!mxIsInt32(Sizes_3D)) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}
	int32_t * const pSizes_3D = static_cast<int32_t *>(mxGetData(Sizes_3D));
	

	const mwSize *dim_array_residues; mwSize dims_residues;
	dims_residues = mxGetNumberOfDimensions(residues);
	dim_array_residues = mxGetDimensions(residues);
	if (dims_residues!=2 || dim_array_residues[0]!=6) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	const mwSize *dim_array_shortest_list; mwSize dims_shortest_list;
	dims_shortest_list = mxGetNumberOfDimensions(shortest_list);
	dim_array_shortest_list = mxGetDimensions(shortest_list);
	if (!(dims_shortest_list == 2 || dims_shortest_list == 3) || dim_array_shortest_list[0] != 2) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	if ((!(dim_array_residues[1] == 1)) && (dim_array_residues[1] != dim_array_shortest_list[2])) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	const mwSize *dim_array_square_size; mwSize dims_square_size;
	dims_square_size = mxGetNumberOfDimensions(square_size);
	dim_array_square_size = mxGetDimensions(square_size);
	if (get_1D_array_size(dims_square_size, dim_array_square_size) == 0) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	const mwSize *dim_array_lookup_table; mwSize dims_lookup_table;
	dims_lookup_table = mxGetNumberOfDimensions(lookup_table);
	dim_array_lookup_table = mxGetDimensions(lookup_table);
	if (get_1D_array_size(dims_lookup_table, dim_array_lookup_table) == 0) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	const mwSize *dim_array_lookup_z_start; mwSize dims_lookup_z_start;
	dims_lookup_z_start = mxGetNumberOfDimensions(lookup_z_start);
	dim_array_lookup_z_start = mxGetDimensions(lookup_z_start);
	if (get_1D_array_size(dims_lookup_z_start, dim_array_lookup_z_start) == 0) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}
	if (get_1D_array_size(dims_lookup_z_start, dim_array_lookup_z_start) != get_1D_array_size(dims_square_size, dim_array_square_size)) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	const mwSize *dim_array_Sizes_3D; mwSize dims_Sizes_3D;
	dims_Sizes_3D = mxGetNumberOfDimensions(Sizes_3D);
	dim_array_Sizes_3D = mxGetDimensions(Sizes_3D);
	mwSize Sizes_3D_1Dsize = get_1D_array_size(dims_Sizes_3D, dim_array_Sizes_3D);
	if (!(Sizes_3D_1Dsize == 2 || Sizes_3D_1Dsize == 3)) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}
	std::array<int,3> size_residue_3D = { (int)pSizes_3D[0],(int)pSizes_3D[1],1 };
	if (Sizes_3D_1Dsize == 3) {
		size_residue_3D[2] = (int)pSizes_3D[2];
	}

	int residue_number = (int)dim_array_residues[1];
	int z_stack_number = get_1D_array_size(dims_lookup_z_start, dim_array_lookup_z_start);
	int lookup_length = get_1D_array_size(dims_lookup_table, dim_array_lookup_table);
	int precomputed_distance_number = (int)dim_array_shortest_list[1];

	int number_of_branches = 0;

	//get the branches
	std::vector<branche_coordinates> returned_branches;
	returned_branches = get_branches_goldstein_compute( presidues, pshortest_list, psquare_size,  plookup_table, plookup_z_start, residue_number, z_stack_number, lookup_length, precomputed_distance_number, size_residue_3D);
	//finish getting branches

	
	number_of_branches = returned_branches.size();

	const int column_number = 6;

	mwSize output_size[2] = { (mwSize)column_number,(mwSize)number_of_branches };


	mxArray * output = mxCreateNumericArray(2, output_size, mxINT32_CLASS, mxREAL);//1D array
	int32_t * poutput = static_cast<int32_t *>(mxGetData(output));

	

	for (int i = 0; i < returned_branches.size(); i++) {//fill up the output array 
		poutput[i*column_number] = returned_branches[i].point_1_x;
		poutput[i*column_number+1] = returned_branches[i].point_1_y;
		poutput[i*column_number+2] = returned_branches[i].point_2_x;
		poutput[i*column_number+3] = returned_branches[i].point_2_y;
		poutput[i*column_number+4] = returned_branches[i].point_z;
		poutput[i*column_number+5] = returned_branches[i].jump_value;
	}

	if (nlhs >= 1) {
		plhs[0] = output;
	}
	if (nlhs >= 2) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	
}

