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
#include "get_shortest_sort_mex.hpp"
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

	if (!(nrhs == 6)) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}

	// We expect to receive as input an initial board, consisting of CPU data of
	// MATLAB class 'logical', and a scalar double specifying the number of
	// generations to compute.

	for (int i = 0; i < nrhs; ++i) {
			if (!mxIsGPUArray(prhs[i])) {
				if (i!=4 && i != 5) mexErrMsgIdAndTxt(errId, errMsg1);
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
	if (dResidue != 2 || sResidue[0] != 6) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	//Phase
	mxGPUArray const * const Lookup = mxGPUCreateFromMxArray(prhs[1]);//because can modify it and return
	mxComplexity const cLookup = mxGPUGetComplexity(Lookup);
	mxClassID const tLookup = mxGPUGetClassID(Lookup);
	mwSize const dLookup = mxGPUGetNumberOfDimensions(Lookup);
	mwSize const * const sLookup = mxGPUGetDimensions(Lookup);
	int32_t const * const pLookup = static_cast<int32_t const *>(mxGPUGetDataReadOnly(Lookup));//because can modify it and return


	//check for the types
	if (tLookup != mxINT32_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg4);
	}
	//check for complexity
	if (cLookup) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	//check for dimentions
	if (get_1D_array_size(dLookup, sLookup) == 0) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	mxGPUArray const * const Lookup_z = mxGPUCreateFromMxArray(prhs[2]);//because can modify it and return
	mxComplexity const cLookup_z = mxGPUGetComplexity(Lookup_z);
	mxClassID const tLookup_z = mxGPUGetClassID(Lookup_z);
	mwSize const dLookup_z = mxGPUGetNumberOfDimensions(Lookup_z);
	mwSize const * const sLookup_z = mxGPUGetDimensions(Lookup_z);
	int32_t const * const pLookup_z = static_cast<int32_t const *>(mxGPUGetDataReadOnly(Lookup_z));//because can modify it and return


	//check for the types
	if (tLookup_z != mxINT32_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg4);
	}
	//check for complexity
	if (cLookup_z) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	//check for dimentions
	if (get_1D_array_size(dLookup_z, sLookup_z) == 0) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	//Phase
	mxGPUArray const * const SquareSize = mxGPUCreateFromMxArray(prhs[3]);//because can modify it and return
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


	mxArray * const Sizes = mxDuplicateArray(prhs[4]);
	const mwSize *dim_array_Sizes; mwSize dims_Sizes;
	dims_Sizes = mxGetNumberOfDimensions(Sizes);
	dim_array_Sizes = mxGetDimensions(Sizes);
	mwSize Sizes_1Dsize = get_1D_array_size(dims_Sizes, dim_array_Sizes);
	if (!(Sizes_1Dsize == 1)) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}
	if (!mxIsInt32(Sizes)) {
		mexErrMsgIdAndTxt(errId, errMsg6);
	}
	if (mxIsComplex(Sizes)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	int32_t * const pSizes = static_cast<int32_t *>(mxGetData(Sizes));

	mxArray * const Sizes_3D = mxDuplicateArray(prhs[5]);
	const mwSize *dim_array_Sizes_3D; mwSize dims_Sizes_3D;
	dims_Sizes_3D = mxGetNumberOfDimensions(Sizes_3D);
	dim_array_Sizes_3D = mxGetDimensions(Sizes_3D);
	mwSize Sizes_3D_1Dsize = get_1D_array_size(dims_Sizes_3D, dim_array_Sizes_3D);
	if (!(Sizes_3D_1Dsize == 2 || Sizes_3D_1Dsize == 3)) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}
	if (!mxIsInt32(Sizes_3D)) {
		mexErrMsgIdAndTxt(errId, errMsg6);
	}
	if (mxIsComplex(Sizes_3D)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	int32_t * const pSizes_3D = static_cast<int32_t *>(mxGetData(Sizes_3D));

	int size_residue_3D[3] = { (int)pSizes_3D[0],(int)pSizes_3D[1],1 };
	if (Sizes_1Dsize == 3) {
		size_residue_3D[2] = (int)pSizes_3D[2];
	}


	int precompute_number = (int)*pSizes;
	int residue_number_1D = sResidue[1];
	mwSize residue_number_1D_mwSize = (mwSize)residue_number_1D;
	mwSize size_output[3] = { (mwSize)2 , (mwSize)precompute_number , residue_number_1D_mwSize };

	int lookup_size = get_1D_array_size(dLookup, sLookup);

	// Create the output matrices

	mxGPUArray *  Nearest_res = mxGPUCreateGPUArray((mwSize)(3), size_output,
		mxINT32_CLASS, mxREAL,
		MX_GPU_DO_NOT_INITIALIZE);

	int32_t * const pNearest_res = static_cast<int32_t *>(mxGPUGetData(Nearest_res));

	

	int res = shortest_sort_KERNEL(pResidue,pLookup,pLookup_z,pSquareSize,pNearest_res,residue_number_1D,precompute_number, size_residue_3D,lookup_size);


	// Wrap the appropriate workspace up as a MATLAB gpuArray for return.
	if (nlhs >= 1) {
		plhs[0] = mxGPUCreateMxArrayOnGPU(Nearest_res);
	}
	if (nlhs > 1) {
		mexErrMsgIdAndTxt(errId, errMsg5);
	}


	// The mxGPUArray pointers are host-side structures that refer to device
	// data. These must be destroyed before leaving the MEX function.
	mxGPUDestroyGPUArray(Residue);
	mxGPUDestroyGPUArray(Lookup);
	mxGPUDestroyGPUArray(Lookup_z);
	mxGPUDestroyGPUArray(SquareSize);
	mxGPUDestroyGPUArray(Nearest_res);
	
}
