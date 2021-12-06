#include"static_parameter.h"

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include <stdint.h>
#include <stdlib.h>
#include <chrono>
#include <atomic>
#include <memory>

#include "CudaBorn.h"
#include "mex_utility.h"

#include <chrono>
#include <thread>

void execute_action(uint32_t action, uint32_t id, std::vector<const mxArray*> rhs, int nlhs, mxArray** plhs);


static std::map<uint32_t, std::shared_ptr<parameter_born>> execution_parameters;

// Mex gateway function
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{

	mxInitGPU();

	if (nrhs < 2) {
		mexErrMsgTxt("Need at least the action and id");
	}

	uint32_t action;
	uint32_t id;


	exact_convert(prhs[0], action);
	exact_convert(prhs[1], id);

	if(display_debug)mexPrintf("ACTION : %i ; ID : %i \n", action, id);

	std::vector<const mxArray*> rhs(prhs + 2, prhs + nrhs);
	int curr_gpu;
	cudaGetDevice(&curr_gpu);
	execute_action(action, id, rhs, nlhs, plhs);
	cudaSetDevice (curr_gpu);
}


void execute_action(uint32_t action, uint32_t id, std::vector<const mxArray*> rhs, int nlhs, mxArray** plhs)
{
	
	switch (action) {
	case 0: //upload the vars (cropped green) + params
	{
		if (display_debug)mexPrintf("Initing the cuda class \n");
		
		int num_input = rhs.size();
		if (num_input < 3) {
			mexErrMsgTxt("Need to give parameters and the two green founction");
		}

		static std::atomic<uint32_t> next_id = {0};
		uint32_t this_id = next_id++;
		if (this_id == UINT32_MAX) {
			next_id--;
			mexErrMsgTxt("Restart the mex the id are all used");
		}
		if (display_debug)mexPrintf("id is %i\n", this_id);
		id = this_id;
		execution_parameters[this_id]= std::make_shared<parameter_born>(); 
		if (display_debug)mexPrintf("parse data\n");
		std::map<std::string, const mxArray*> map_input;
		convert(rhs[0], map_input);
		search_and_set(map_input, "simulation_size", execution_parameters[id]->simulation_size);
		search_and_set(map_input, "RI_size", execution_parameters[id]->RI_size);
		search_and_set(map_input, "field_size", execution_parameters[id]->field_size);
		search_and_set(map_input, "conv_size", execution_parameters[id]->conv_size);
		search_and_set(map_input, "green_size", execution_parameters[id]->green_size);
		if (display_debug)mexPrintf("parse gpu\n");
		search_and_set(map_input, "used_gpu", execution_parameters[id]->used_gpu);
        if (display_debug)mexPrintf("check gpu\n");
        execution_parameters[id]->check_GPU();
		if (display_debug)mexPrintf("Setup cuda, vars \n");
		execution_parameters[id]->create_cuda_data();
		if (display_debug)mexPrintf("Setup fft plans \n");
		execution_parameters[id]->prepare_fft3();
		execution_parameters[id]->prepare_fft2();
		if (display_debug)mexPrintf("Setup green function \n");
		execution_parameters[id]->set_green(rhs[1]);
		execution_parameters[id]->set_green_2(rhs[2]);

		if (nlhs > 0) {
			convert((uint32_t)this_id, plhs[0]);
		}

		break;
	}
	case 1:// upload the RI
	{
		if (display_debug)mexPrintf("Upload refractive index \n");
		
		if (execution_parameters.find(id) == execution_parameters.end()) {
			mexErrMsgTxt("Id not found");
		}

		int num_input = rhs.size();
		if (num_input < 2) {
			mexErrMsgTxt("Need to give both parameters and scattering potential");
		}

		std::map<std::string, const mxArray*> map_input;
		//mexPrintf("1\n");
		convert(rhs[0], map_input);
		search_and_set(map_input, "k0_nm", execution_parameters[id]->k0_nm);
		//mexPrintf("2\n");
		search_and_set(map_input, "eps_imag", execution_parameters[id]->eps_imag);
		//mexPrintf("3\n");
		search_and_set(map_input, "fourier_res1", execution_parameters[id]->fourier_res[0]);
		//mexPrintf("4\n");
		search_and_set(map_input, "fourier_res2", execution_parameters[id]->fourier_res[1]);
		//mexPrintf("5\n");
		search_and_set(map_input, "fourier_res3", execution_parameters[id]->fourier_res[2]);
		//mexPrintf("6\n");
		search_and_set(map_input, "ROI", execution_parameters[id]->ROI_simulation);
		//mexPrintf("7\n");
		search_and_set(map_input, "Bornmax", execution_parameters[id]->Bornmax);
		//mexPrintf("8\n");
		search_and_set(map_input, "dV", execution_parameters[id]->dV);
		//mexPrintf("9\n");
		execution_parameters[id]->set_V(rhs[1]);
		//mexPrintf("10\n");

		break;
	}
	case 2://solve
	{
		if (display_debug)mexPrintf("Solving \n");

		if (execution_parameters.find(id) == execution_parameters.end()) {
			mexErrMsgTxt("Id not found");
		}

		if (display_debug)execution_parameters[id]->display_state();

		int num_input = rhs.size();
		if (num_input < 1) {
			mexErrMsgTxt("Need to give the field to simulate");
		}
		if (display_debug)mexPrintf("Upload field \n");
		execution_parameters[id]->set_source_3D(rhs[0]);
		if (display_debug)mexPrintf("execute \n");
		execution_parameters[id]->execute();
		if (nlhs > 0) {
			if (display_debug)mexPrintf("get result \n");
			execution_parameters[id]->get_field_3D(plhs[0]);
		}

		break;
	}
	case 3://clean up the gpu memory
	{
		if (display_debug)mexPrintf("Clean cuda memory \n");

		if (execution_parameters.find(id) == execution_parameters.end()) {
			mexErrMsgTxt("Id not found");
		}

		execution_parameters.erase(id);
		break;
	}
	case 4://solve v2
	{
		if (display_debug)mexPrintf("Solving \n");

		if (execution_parameters.find(id) == execution_parameters.end()) {
			mexErrMsgTxt("Id not found");
		}

		if (display_debug)execution_parameters[id]->display_state();

		int num_input = rhs.size();
		if (num_input < 2) {
			mexErrMsgTxt("Need to give the field to simulate and the gpu to use");
		}
		std::vector<uint32_t> gpu_vect;
		convert(rhs[1], gpu_vect);
		execution_parameters[id]->loaded_gpu=std::vector<size_t>(gpu_vect.begin(), gpu_vect.end());
		if (display_debug)mexPrintf("Upload field \n");
		execution_parameters[id]->set_source_2D(rhs[0]);
		if (display_debug)mexPrintf("execute \n");
		execution_parameters[id]->execute_2D_in();
		execution_parameters[id]->execute();
		execution_parameters[id]->execute_2D_out();
		if (nlhs > 0) {
			if (display_debug)mexPrintf("get result \n");
			execution_parameters[id]->get_trans_2D(plhs[0]);
			if (nlhs > 1) {
				execution_parameters[id]->get_ref_2D(plhs[1]);
				if (nlhs > 2) {
					execution_parameters[id]->get_field_3D(plhs[2]);
				}
			}
		}

		break;
	}
	}
	
}
int mexAtExit(void (*ExitFcn)(void)) {
	// releases all memmory
	if (display_debug)mexPrintf("Quit cuda mex \n");
	execution_parameters.clear();
	return 0;
}