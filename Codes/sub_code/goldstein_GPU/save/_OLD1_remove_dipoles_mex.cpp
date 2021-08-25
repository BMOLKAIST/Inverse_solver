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
#include "remove_dipoles_mex.hpp"
#include <algorithm>

 /**
  * MEX gateway
  */
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{

	//INPUTS : Rythov field (the fields) ; Single fourier transform (precomputed ewald shpere) ; exp_Phase (precomputed exponential factors) ; shifts (angle for the positions)

	char const * const errId = "parallel:gpu:pctdemo_life_mex:InvalidInput";
	char const * const errMsg = "Provide the residue and positive residues positions as a GPU array real single type of dimention 2 or 3.";
	char const * const errMsg1 = "Provide the residue and positive residues positions as a GPU array";
	char const * const errMsg2 = "Provide the residue and positive residues positions of dimention 2 or 3.";
	char const * const errMsg3 = "Provide the residue and positive residues positions as real.";
	char const * const errMsg4 = "Provide the residue as single ";
	char const * const errMsg5 = "Need two outputs";
	char const * const errMsg6 = "Provide the  positive residues positions as int32";
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
				mexErrMsgIdAndTxt(errId, errMsg1);
			}
	}


	//Phase
	mxGPUArray * const Residue = mxGPUCopyFromMxArray(prhs[0]);//because can modify it and return
	mxComplexity const cResidue = mxGPUGetComplexity(Residue);
	mxClassID const tResidue = mxGPUGetClassID(Residue);
	mwSize const dResidue = mxGPUGetNumberOfDimensions(Residue);
	mwSize const * const sResidue = mxGPUGetDimensions(Residue);
	float* const pResidue = static_cast<float*>(mxGPUGetData(Residue));//because can modify it and return
	
	
	//check for the types
	if (tResidue != mxSINGLE_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg4);
	}
	//check for complexity
	if (cResidue) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	//check for dimentions
	if (!(dResidue == 3 || dResidue == 2)) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}


	mxGPUArray * const Positive = mxGPUCopyFromMxArray(prhs[1]);//because can modify it and return
	mxComplexity const cPositive = mxGPUGetComplexity(Positive);
	mxClassID const tPositive = mxGPUGetClassID(Positive);
	mwSize const dPositive = mxGPUGetNumberOfDimensions(Positive);
	mwSize const * const sPositive = mxGPUGetDimensions(Positive);
	int32_t* const pPositive = static_cast<int32_t*>(mxGPUGetData(Positive));//because can modify it and return

	//check for the types
	if (tPositive != mxINT32_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg6);
	}
	//check for complexity
	if (cPositive) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	//check for dimentions
	if (!(dPositive == 1 || dPositive == 2)) {
		if (dPositive == 2 && !(sPositive[0] == 1 || sPositive[1] == 1)) {
			mexErrMsgIdAndTxt(errId, errMsg2);
		}
	}

	// Create two workspace gpuArrays, square real arrays of the same size as
	// the input containing logical data. We will fill these with data, so leave
	// them uninitialized.
	mwSize output_size[3] = { ((mwSize)(sResidue[0])+1), ((mwSize)(sResidue[1])+1), (mwSize)(1) }; //residue is one smaller than original

	if (dResidue == 3) {
		output_size[2] = sResidue[2];
	}

	mwSize output2_size[2] = {output_size[0],output_size[2]}; //residue is one smaller than original


	mxGPUArray *  outArray = mxGPUCreateGPUArray((mwSize)(3), output_size,
		mxSINGLE_CLASS, mxREAL,
		MX_GPU_INITIALIZE_VALUES);

	float * const pOutArray = static_cast<float *>(mxGPUGetData(outArray));

	mxGPUArray *  out2Array = mxGPUCreateGPUArray((mwSize)(2), output2_size,
		mxSINGLE_CLASS, mxREAL,
		MX_GPU_INITIALIZE_VALUES);

	float * const pOut2Array = static_cast<float *>(mxGPUGetData(outArray));

	int dims_output[3] = { (size_t)(output_size[0]) ,(size_t)(output_size[1]),(size_t)(output_size[2]) };

	int dims_Positive =  (size_t)(sPositive[0]) ;
	if (dPositive == 2 && sPositive[0] == 1) {
		dims_Positive = (size_t)(sPositive[1]);
	}
	

	int res = dipoles_KERNEL(pResidue, pPositive, pOutArray, pOutArray, dims_output, dims_Positive);


	// Wrap the appropriate workspace up as a MATLAB gpuArray for return.
	if (nlhs >= 1) {
		plhs[0] = mxGPUCreateMxArrayOnGPU(Residue);
	}
	if (nlhs >= 2) {
		plhs[1] = mxGPUCreateMxArrayOnGPU(outArray);
	}
	if (nlhs >= 3) {
		plhs[2] = mxGPUCreateMxArrayOnGPU(out2Array);
	}
	if (nlhs > 3) {
		mexErrMsgIdAndTxt(errId, errMsg5);
	}


	// The mxGPUArray pointers are host-side structures that refer to device
	// data. These must be destroyed before leaving the MEX function.
	mxGPUDestroyGPUArray(Positive);
	mxGPUDestroyGPUArray(Residue);
	mxGPUDestroyGPUArray(outArray);
	mxGPUDestroyGPUArray(out2Array);
	
}
