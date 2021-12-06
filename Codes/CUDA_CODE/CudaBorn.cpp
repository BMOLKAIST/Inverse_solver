#include"static_parameter.h"
#include"CudaBorn.h"
#include"memory_management_cuda.h"
#include <mex.h>

parameter_born::parameter_born() {

}
parameter_born::~parameter_born() {
	delete_cuda_data();
}
void parameter_born::check_GPU(){
    if(used_gpu.size()==0){
        used_gpu=std::vector<uint32_t>({0});
    }
	if (display_debug_2){
		mexPrintf("GPU list : \n");
		for(auto gpu:used_gpu){
			mexPrintf("   %d \n",gpu);
		}
	}
}
void parameter_born::set_V(mxArray const* const matlab_data) {
	output_ready = false;
	scattering_potential->set_data_multi_duplicate_all(matlab_data);
}
void parameter_born::set_source_3D(mxArray const* const matlab_data) {
	output_ready = false;
	var1->set_data_multi(matlab_data,loaded_gpu);
}
void parameter_born::set_source_2D(mxArray const* const  matlab_data) {
	output_ready = false;
	source_2D->set_data_multi(matlab_data,loaded_gpu);
}
void parameter_born::set_green(mxArray const* const  matlab_data) {
	output_ready = false;
	free_space_green->set_data_multi_duplicate_all(matlab_data);
}
void parameter_born::set_green_2(mxArray const* const  matlab_data) {
	output_ready = false;
	free_space_green_2->set_data_multi_duplicate_all(matlab_data);
}
void parameter_born::get_field_3D(mxArray*& matlab_data) {
	if (!output_ready) {
		//mexErrMsgTxt("Output is not ready");
	}

	std::vector<size_t> RI_sz(RI_size.begin(), RI_size.end());
	var1->get_new_data_multi(matlab_data,RI_sz,loaded_gpu);
}
void parameter_born::get_trans_2D(mxArray*& matlab_data) {
	if (!output_ready) {
		//mexErrMsgTxt("Output is not ready");
	}
	trans_2D->get_new_data_multi(matlab_data,loaded_gpu);
}
void parameter_born::get_ref_2D(mxArray*& matlab_data) {
	if (!output_ready) {
		//mexErrMsgTxt("Output is not ready");
	}
	ref_2D->get_new_data_multi(matlab_data,loaded_gpu);
	//temp_fft2->get_new_data(matlab_data);
	//free_space_green_2->get_new_data(matlab_data);
}
void parameter_born::display_state() {
	mexPrintf("Parameter born content : \n");
	mexPrintf("    eps_imag : %f\n", eps_imag);
	mexPrintf("    k0_nm : %f\n", k0_nm);
	mexPrintf("    fourier res : \n");
	for (auto sz : fourier_res) {
		mexPrintf("        %f\n", sz);
	}
	mexPrintf("    simulation size : \n");
	for (auto sz : simulation_size) {
		mexPrintf("        %i\n", sz);
	}
	mexPrintf("    RI size : \n");
	for (auto sz : RI_size) {
		mexPrintf("        %i\n", sz);
	}
	mexPrintf("    field size : \n");
	for (auto sz : field_size) {
		mexPrintf("        %i\n", sz);
	}
	mexPrintf("    ROI : \n");
	for (auto sz : ROI_simulation) {
		mexPrintf("        %i\n", sz);
	}
}
void parameter_born::create_cuda_data() {
	

	delete_cuda_data();
	std::vector<size_t> size_init(simulation_size.begin(), simulation_size.end());
	var1 = std::make_shared<CUDA_ARRAY<float2>>(size_init,used_gpu,&curr_gpu);
	var2 = std::make_shared<CUDA_ARRAY<float2>>(size_init,used_gpu,&curr_gpu);
	var3 = std::make_shared<CUDA_ARRAY<float2>>(size_init,used_gpu,&curr_gpu);
	scattering_potential = std::make_shared<CUDA_ARRAY<float2>>(size_init,used_gpu,&curr_gpu);

	std::vector<size_t> gr_sz(green_size.begin(), green_size.end());
	std::vector<size_t> fl_sz(field_size.begin(), field_size.end());
	std::vector<size_t> cv_sz(conv_size.begin(), conv_size.end());
	std::vector<size_t> cv_2D_sz(conv_size.begin(), conv_size.end());
	cv_2D_sz[2] = 1;

	free_space_green = std::make_shared<CUDA_ARRAY<float2>>(gr_sz,used_gpu,&curr_gpu);
	free_space_green_2 = std::make_shared<CUDA_ARRAY<float2>>(gr_sz,used_gpu,&curr_gpu);
	temp_fft2 = std::make_shared<CUDA_ARRAY<float2>>(cv_sz,used_gpu,&curr_gpu);
	source_2D = std::make_shared<CUDA_ARRAY<float2>>(fl_sz,used_gpu,&curr_gpu);
	trans_2D = std::make_shared<CUDA_ARRAY<float2>>(fl_sz,used_gpu,&curr_gpu);
	ref_2D = std::make_shared<CUDA_ARRAY<float2>>(fl_sz,used_gpu,&curr_gpu);
	temp_field = std::make_shared<CUDA_ARRAY<float2>>(cv_2D_sz,used_gpu,&curr_gpu);
	temp_field_2 = std::make_shared<CUDA_ARRAY<float2>>(cv_2D_sz,used_gpu,&curr_gpu);
}
void parameter_born::delete_cuda_data() {
	
	var1.reset();
	var2.reset();
	var3.reset();

	free_space_green.reset();
	free_space_green_2.reset();
	temp_fft2.reset();

	source_2D.reset();
	trans_2D.reset();
	ref_2D.reset();

	temp_field.reset();
	temp_field_2.reset();

	scattering_potential.reset();
	

	destroy_fft3();
	destroy_fft2();
}
