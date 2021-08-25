/**
 * @file pctdemo_life_mex_texture.cpp
 * @brief MEX gateway for a stencil operation.
 * Copyright 2013 The MathWorks, Inc.
 *
 */

#include <stdint.h>
#include "tmwtypes.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "get_lookup_mex.hpp"
#include <algorithm>

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

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{

	//INPUTS : Rythov field (the fields) ; Single fourier transform (precomputed ewald shpere) ; exp_Phase (precomputed exponential factors) ; shifts (angle for the positions)

	char const * const errId = "parallel:gpu:pctdemo_life_mex:InvalidInput";
	char const * const errMsg = "Provide the residues and residues and square size size as a GPU array real single type of dimention 1.";
	char const * const errMsg1 = "Provide the residues and residues and square size size as a GPU array";
	char const * const errMsg2 = "Provide the residues and square size size of dimention 1.";
	char const * const errMsg3 = "Provide the residues and residues and square size size as real.";
	char const * const errMsg4 = "Provide the residues and square size as int32 ";
	char const * const errMsg5 = "Need two outputs";
	char const * const errMsg6 = "Provide the residues size as int32";
	char const * const errMsg7 = "Provide the residues size must be X*6";
	// Initialize the MathWorks GPU API.
	mxInitGPU();

	if (!(nrhs == 2)) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}

	// We expect to receive as input an initial board, consisting of CPU data of
	// MATLAB class 'logical', and a scalar double specifying the number of
	// generations to compute.

	for (int i = 0; i < nrhs; ++i) {
			if (!mxIsGPUArray(prhs[i])) {
				if (i!=2) mexErrMsgIdAndTxt(errId, errMsg1);
			}
	}


	//Phase
	mxGPUArray const * const Residue = mxGPUCreateFromMxArray(prhs[0]);//because can modify it and return
	mxComplexity const cResidue = mxGPUGetComplexity(Residue);
	mxClassID const tResidue = mxGPUGetClassID(Residue);
	mwSize const dResidue = mxGPUGetNumberOfDimensions(Residue);
	mwSize const * const sResidue = mxGPUGetDimensions(Residue);
	int32_t const * const pResidue = static_cast<int32_t const *>(mxGPUGetDataReadOnly(Residue));//because can modify it and return
	
	
	//check for the types
	if (tResidue != mxINT32_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg4);
	}
	//check for complexity
	if (cResidue) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	//check for dimentions
	if (dResidue!=2 || sResidue[0] != 6) {
		mexErrMsgIdAndTxt(errId, errMsg7);
	}

	//Phase
	mxGPUArray const * const SquareSize = mxGPUCreateFromMxArray(prhs[1]);//because can modify it and return
	mxComplexity const cSquareSize = mxGPUGetComplexity(SquareSize);
	mxClassID const tSquareSize = mxGPUGetClassID(SquareSize);
	mwSize const dSquareSize = mxGPUGetNumberOfDimensions(SquareSize);
	mwSize const * const sSquareSize = mxGPUGetDimensions(SquareSize);
	int32_t const * const pSquareSize = static_cast<int32_t const *>(mxGPUGetDataReadOnly(SquareSize));//because can modify it and return

	

	//check for the types
	if (tSquareSize != mxINT32_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg4);
	}
	//check for complexity
	if (cSquareSize) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	//check for dimentions
	if (get_1D_array_size(dSquareSize, sSquareSize) == 0) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	
	// Create the output matrices

	mxArray * const SquareSize_CPU = mxGPUCreateMxArrayOnCPU(SquareSize);
	int32_t const * const pSquareSize_CPU = static_cast<int32_t const *>(mxGetData(SquareSize_CPU));

	int size_lookup_table = 0;
	int size_residue_1D = (int)sResidue[1];
	int size_square_1D = (int)get_1D_array_size(dSquareSize, sSquareSize);

	mwSize size_square_1D_mwsize = (mwSize)size_square_1D;

	mxArray * lookup_z_start_CPU = mxCreateNumericArray(1, &(size_square_1D_mwsize), mxINT32_CLASS, mxREAL);//1D array
	int32_t * plookup_z_start_CPU = static_cast<int32_t *>(mxGetData(lookup_z_start_CPU));

	for (int i = 0; i < size_square_1D; i++) {
		int z_case_num = pSquareSize_CPU[i];
		plookup_z_start_CPU[i]= size_lookup_table;
		size_lookup_table += z_case_num * z_case_num;
	}

	mxGPUArray const * const lookup_z_start = mxGPUCreateFromMxArray(lookup_z_start_CPU);
	int32_t const * const plookup_z_start = static_cast<int32_t const *>(mxGPUGetDataReadOnly(lookup_z_start));

	mxDestroyArray(lookup_z_start_CPU);//destroy
	mxDestroyArray(SquareSize_CPU);//destroy

	mwSize size_residue_1D_mwSize = (mwSize)size_residue_1D;

	mwSize size_lookup_table_mwsize = (mwSize)size_lookup_table;

	mxGPUArray *  LookupTable = mxGPUCreateGPUArray((mwSize)(1), &size_lookup_table_mwsize,
		mxINT32_CLASS, mxREAL,
		MX_GPU_DO_NOT_INITIALIZE);

	int32_t * const pLookupTable = static_cast<int32_t *>(mxGPUGetData(LookupTable));

	int dim_1_residue = sResidue[1];
	

	int res = lookup_KERNEL(pLookupTable, plookup_z_start, pResidue, dim_1_residue, size_lookup_table);


	// Wrap the appropriate workspace up as a MATLAB gpuArray for return.
	if (nlhs >= 1) {
		plhs[0] = mxGPUCreateMxArrayOnGPU(LookupTable);
	}
	if (nlhs >= 2) {
		plhs[1] = mxGPUCreateMxArrayOnGPU(lookup_z_start);
	}
	if (nlhs > 2) {
		mexErrMsgIdAndTxt(errId, errMsg5);
	}


	// The mxGPUArray pointers are host-side structures that refer to device
	// data. These must be destroyed before leaving the MEX function.
	
	mxGPUDestroyGPUArray(LookupTable);
	mxGPUDestroyGPUArray(lookup_z_start);
	mxGPUDestroyGPUArray(Residue);
	mxGPUDestroyGPUArray(SquareSize);
	
}
