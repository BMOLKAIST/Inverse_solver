
#include <stdint.h>
#include "tmwtypes.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "unwrapp_phase_goldstein_mex.hpp"
#include <algorithm>
#include <array>

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

	if (!(nrhs == 3)) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}


	for (int i = 0; i < nrhs; ++i) {
			if (!mxIsGPUArray(prhs[i])) {
				mexErrMsgIdAndTxt(errId, errMsg1);
			}
	}


	//Phase
	mxGPUArray * const Phase = mxGPUCopyFromMxArray(prhs[0]);//because can modify it and return
	mxComplexity const cPhase = mxGPUGetComplexity(Phase);
	mxClassID const tPhase = mxGPUGetClassID(Phase);
	mwSize const dPhase = mxGPUGetNumberOfDimensions(Phase);
	mwSize const * const sPhase = mxGPUGetDimensions(Phase);
	float * const pPhase = static_cast<float *>(mxGPUGetData(Phase));//because can modify it and return
	
	
	//check for the types
	if (tPhase != mxSINGLE_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg4);
	}
	//check for complexity
	if (cPhase) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	//check for dimentions
	if (!(dPhase ==2 || dPhase == 3)) {
		mexErrMsgIdAndTxt(errId, errMsg7);
	}

	//Phase
	mxGPUArray const * const Step1 = mxGPUCreateFromMxArray(prhs[1]);//because can modify it and return
	mxComplexity const cStep1 = mxGPUGetComplexity(Step1);
	mxClassID const tStep1 = mxGPUGetClassID(Step1);
	mwSize const dStep1 = mxGPUGetNumberOfDimensions(Step1);
	mwSize const * const sStep1 = mxGPUGetDimensions(Step1);
	int32_t const * const pStep1 = static_cast<int32_t const *>(mxGPUGetDataReadOnly(Step1));//because can modify it and return


	//check for the types
	if (tStep1 != mxINT32_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg4);
	}
	//check for complexity
	if (cStep1) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	//check for dimentions
	if (!(dStep1 == 3 || dStep1 == 2)) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	mxGPUArray const * const Step2 = mxGPUCreateFromMxArray(prhs[2]);//because can modify it and return
	mxComplexity const cStep2 = mxGPUGetComplexity(Step2);
	mxClassID const tStep2 = mxGPUGetClassID(Step2);
	mwSize const dStep2 = mxGPUGetNumberOfDimensions(Step2);
	mwSize const * const sStep2 = mxGPUGetDimensions(Step2);
	int32_t const * const pStep2 = static_cast<int32_t const *>(mxGPUGetDataReadOnly(Step2));//because can modify it and return


	//check for the types
	if (tStep2 != mxINT32_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg4);
	}
	//check for complexity
	if (cStep2) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	//check for dimentions
	if (!(dStep2 <= 2)) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}
	
	std::array<int, 3> size_Phase = { (int)sPhase[0],(int)sPhase[1],1 };
	std::array<int, 3> size_Step1 = { (int)sStep1[0],(int)sStep1[1],1 };
	std::array<int, 2> size_Step2 = { 1,1 };
	if (get_1D_array_size(dStep2, sStep2) != 0) {
		size_Step2[0] = get_1D_array_size(dStep2, sStep2);
	}
	else {
		size_Step2[0] = (int)sStep2[0];
		size_Step2[1] = (int)sStep2[1];
	}
	if (dStep1 == 3) {
		size_Step1[2] = (int)sStep1[2];
	}
	if (dPhase == 3) {
		size_Phase[2] = (int)sPhase[2];
	}
	
	if (size_Step2[0]!= size_Step1[1] || size_Step2[1] != size_Step1[2]) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}

	int res = unwrapp_KERNEL(pPhase, pStep1, pStep2, size_Phase);


	// Wrap the appropriate workspace up as a MATLAB gpuArray for return.
	if (nlhs >= 1) {
		plhs[0] = mxGPUCreateMxArrayOnGPU(Phase);
	}
	if (nlhs > 1) {
		mexErrMsgIdAndTxt(errId, errMsg5);
	}


	// The mxGPUArray pointers are host-side structures that refer to device
	// data. These must be destroyed before leaving the MEX function.
	
	mxGPUDestroyGPUArray(Phase);
	mxGPUDestroyGPUArray(Step1);
	mxGPUDestroyGPUArray(Step2);
	
}
