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
#include "create_sort_value_mex.hpp"
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
	char const * const errMsg2 = "Provide the residues and residues and square size size of dimention 1.";
	char const * const errMsg3 = "Provide the residues and residues and square size size as real.";
	char const * const errMsg4 = "Provide the residues and square size as int32 ";
	char const * const errMsg5 = "Need two outputs";
	char const * const errMsg6 = "Provide the residues size as int32";
	// Initialize the MathWorks GPU API.
	mxInitGPU();

	if (!(nrhs == 3)) {
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
	if (get_1D_array_size(dResidue, sResidue) == 0) {
		mexErrMsgIdAndTxt(errId, errMsg2);
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

	mxArray * const Sizes = mxDuplicateArray(prhs[2]);
	const mwSize *dim_array_Sizes; mwSize dims_Sizes;
	dims_Sizes = mxGetNumberOfDimensions(Sizes);
	dim_array_Sizes = mxGetDimensions(Sizes);
	mwSize Sizes_1Dsize = get_1D_array_size(dims_Sizes, dim_array_Sizes);
	if (!(Sizes_1Dsize == 2 || Sizes_1Dsize == 3)) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}
	if (!mxIsInt32(Sizes)) {
		mexErrMsgIdAndTxt(errId, errMsg6);
	}
	if (mxIsComplex(Sizes)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	int32_t * const pSizes = static_cast<int32_t *>(mxGetData(Sizes));

	
	// Create the output matrices


	int size_residue_3D[3] = { (int)pSizes[0],(int)pSizes[1],1 };
	if (Sizes_1Dsize == 3) {
		size_residue_3D[2] = (int)pSizes[2];
	}
	int size_residue_1D = get_1D_array_size(dResidue, sResidue);

	mwSize size_residue_1D_mwSize = (mwSize)size_residue_1D;

	mxGPUArray *  CaseID = mxGPUCreateGPUArray((mwSize)(1), &size_residue_1D_mwSize,
		mxINT32_CLASS, mxREAL,
		MX_GPU_DO_NOT_INITIALIZE);

	int32_t * const pCaseID = static_cast<int32_t *>(mxGPUGetData(CaseID));

	mxGPUArray *  Dim1ID = mxGPUCreateGPUArray((mwSize)(1), &size_residue_1D_mwSize,
		mxINT32_CLASS, mxREAL,
		MX_GPU_DO_NOT_INITIALIZE);

	int32_t * const pDim1ID = static_cast<int32_t *>(mxGPUGetData(Dim1ID));

	mxGPUArray *  Dim2ID = mxGPUCreateGPUArray((mwSize)(1), &size_residue_1D_mwSize,
		mxINT32_CLASS, mxREAL,
		MX_GPU_DO_NOT_INITIALIZE);

	int32_t * const pDim2ID = static_cast<int32_t *>(mxGPUGetData(Dim2ID));

	
	

	int res = create_sort_KERNEL(pResidue, pSquareSize, pCaseID, pDim1ID , pDim2ID, size_residue_3D, size_residue_1D);


	// Wrap the appropriate workspace up as a MATLAB gpuArray for return.
	if (nlhs >= 1) {
		plhs[0] = mxGPUCreateMxArrayOnGPU(CaseID);
	}
	if (nlhs >= 2) {
		plhs[1] = mxGPUCreateMxArrayOnGPU(Dim1ID);
	}
	if (nlhs >= 3) {
		plhs[2] = mxGPUCreateMxArrayOnGPU(Dim2ID);
	}
	if (nlhs > 3) {
		mexErrMsgIdAndTxt(errId, errMsg5);
	}


	// The mxGPUArray pointers are host-side structures that refer to device
	// data. These must be destroyed before leaving the MEX function.
	mxGPUDestroyGPUArray(CaseID);
	mxGPUDestroyGPUArray(Dim1ID);
	mxGPUDestroyGPUArray(Dim2ID);
	mxGPUDestroyGPUArray(Residue);
	mxGPUDestroyGPUArray(SquareSize);
	
}
