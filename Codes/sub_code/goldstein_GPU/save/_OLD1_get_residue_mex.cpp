/**
 * @file pctdemo_life_mex_texture.cpp
 * @brief MEX gateway for a stencil operation.
 * Copyright 2013 The MathWorks, Inc.
 *
 */

#include "tmwtypes.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "get_residue_mex.hpp"

 /**
  * MEX gateway
  */
void mexFunction(int /* nlhs */, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{

	//INPUTS : Rythov field (the fields) ; Single fourier transform (precomputed ewald shpere) ; exp_Phase (precomputed exponential factors) ; shifts (angle for the positions)

	char const * const errId = "parallel:gpu:pctdemo_life_mex:InvalidInput";
	char const * const errMsg = "Provdie the phase as a GPU array real single type of dimention 2 or 3.";
	char const * const errMsg1 = "Provdie the phase as a GPU array";
	char const * const errMsg2 = "Provdie the phase of dimention 2 or 3.";
	char const * const errMsg3 = "Provdie the phase as real.";
	char const * const errMsg4 = "Provdie the phase as single.";
	// Initialize the MathWorks GPU API.
	mxInitGPU();

	if (nrhs != 1) {
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
	mxGPUArray const * const Phase = mxGPUCreateFromMxArray(prhs[0]);
	mxComplexity const cPhase = mxGPUGetComplexity(Phase);
	mxClassID const tPhase = mxGPUGetClassID(Phase);
	mwSize const dPhase = mxGPUGetNumberOfDimensions(Phase);
	mwSize const * const sPhase = mxGPUGetDimensions(Phase);
	float const * const pPhase = static_cast<float const *>(mxGPUGetDataReadOnly(Phase));
	
	//check for the types
	if (tPhase != mxSINGLE_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg4);
	}

	//check for complexity
	if (cPhase) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	//check for dimentions
	if (!(dPhase == 3 || dPhase == 2)) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	// Create two workspace gpuArrays, square real arrays of the same size as
	// the input containing logical data. We will fill these with data, so leave
	// them uninitialized.
	mwSize output_size[3] = { ((mwSize)(sPhase[0])-1), ((mwSize)(sPhase[1])-1), (mwSize)(1) }; //residue is one smaller than original

	if (dPhase == 3) {
		output_size[2] = sPhase[2];
	}

	mxGPUArray *  outArray = mxGPUCreateGPUArray((mwSize)(3), output_size,
		mxSINGLE_CLASS, mxREAL,
		MX_GPU_DO_NOT_INITIALIZE);

	float * const pOutArray = static_cast<float *>(mxGPUGetData(outArray));

	int dims_output[3] = { (size_t)(output_size[0]) ,(size_t)(output_size[1]),(size_t)(output_size[2]) };

	

	int res = residue_KERNEL(pPhase, pOutArray, dims_output);


	// Wrap the appropriate workspace up as a MATLAB gpuArray for return.

	plhs[0] = mxGPUCreateMxArrayOnGPU(outArray);

	// The mxGPUArray pointers are host-side structures that refer to device
	// data. These must be destroyed before leaving the MEX function.
	mxGPUDestroyGPUArray(Phase);
	mxGPUDestroyGPUArray(outArray);
}
