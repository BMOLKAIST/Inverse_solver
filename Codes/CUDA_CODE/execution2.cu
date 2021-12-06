#include"static_parameter.h"
#include"CudaBorn.h"
#include"memory_management_cuda.h"
#include <cufft.h>
#include <mex.h>
#include "cuComplex.h"
#include <complex.h>
#include "math_constants.h"
#include <algorithm>

void parameter_born::prepare_fft2() {
	int curr_gpu;
	cudaGetDevice(&curr_gpu);
	destroy_fft2();
	fft_2D_size = std::vector<int>(conv_size.begin(), conv_size.end());
	auto cufft_adapt_size = fft_2D_size;
	cufft_adapt_size.pop_back();
	std::reverse(cufft_adapt_size.begin(), cufft_adapt_size.end());
	int batch_number = fft_2D_size[2];
	int linear_size_2D = std::accumulate(begin(cufft_adapt_size), end(cufft_adapt_size), 1, std::multiplies<int>());
	/* Create a 3D FFT plan. */
	fft_2D_plan=std::vector<cufftHandle>(used_gpu.size());
	for (int gpu=0; gpu<used_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[gpu]);
		if (cufftPlanMany(&fft_2D_plan[gpu], cufft_adapt_size.size(), cufft_adapt_size.data(),
			NULL, 1, linear_size_2D, // *inembed, istride, idist 
			NULL, 1, linear_size_2D, // *onembed, ostride, odist
			CUFFT_C2C, batch_number) != CUFFT_SUCCESS) {
			throw(std::runtime_error("parameter_born::prepare_fft:: CUFFT error: Plan creation failed"));
			return;
		}
	}
	cudaSetDevice (curr_gpu);
	fft_2D_ready = true;
}

void parameter_born::destroy_fft2() {
	int curr_gpu;
	cudaGetDevice(&curr_gpu);
	for (int gpu=0; gpu<used_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[gpu]);
		if (fft_2D_ready){
			cufftDestroy(fft_2D_plan[gpu]);
			if (display_debug)mexPrintf("fft2 deleted\n");
		}else{
			if (display_debug)mexPrintf("fft2 not deleted\n");
		}
	}
	cudaSetDevice (curr_gpu);
	fft_2D_ready = false;
}
void parameter_born::fft2(std::shared_ptr<CUDA_ARRAY<float2>> var){
	fft2(var,curr_gpu);
}
void parameter_born::fft2(std::shared_ptr<CUDA_ARRAY<float2>> var, size_t gpu) {
	cudaSetDevice(used_gpu[gpu]);
	if (!fft_2D_ready) throw(std::runtime_error("parameter_born::fft3::fft is not ready"));
	std::vector<size_t> fft_size_size_t(fft_2D_size.begin(), fft_2D_size.end());
	if (fft_size_size_t != var->get_size()) throw(std::runtime_error("parameter_born::fft3::size not matching"));
	if (cufftExecC2C(fft_2D_plan[gpu], var->cuda_ptr(gpu), var->cuda_ptr(gpu), CUFFT_FORWARD) != CUFFT_SUCCESS) {
		throw(std::runtime_error("parameter_born::fft3::CUFFT error: ExecC2C Forward failed"));
		return;
	}
}
void parameter_born::ifft2(std::shared_ptr<CUDA_ARRAY<float2>> var){
	ifft2(var,curr_gpu);
}
void parameter_born::ifft2(std::shared_ptr<CUDA_ARRAY<float2>> var, size_t gpu) {
	cudaSetDevice(used_gpu[gpu]);
	if (!fft_2D_ready) throw(std::runtime_error("parameter_born::ifft3::ifft is not ready"));
	std::vector<size_t> fft_size_size_t(fft_2D_size.begin(), fft_2D_size.end());
	if (fft_size_size_t != var->get_size()) throw(std::runtime_error("parameter_born::fft3::size not matching"));
	if (cufftExecC2C(fft_2D_plan[gpu], var->cuda_ptr(gpu), var->cuda_ptr(gpu), CUFFT_INVERSE) != CUFFT_SUCCESS) {
		throw(std::runtime_error("parameter_born::fft3::CUFFT error: ExecC2C Forward failed"));
		return;
	}
}

struct data_transfer_params {
	int arrayCount;
	int rect_source[3];
	int rect_destination[3];
	int rect_size[3];
	int size_source[3];
	int size_destination[3];
};
__device__ int modulo_cpy(int m, int n) { return m >= 0 ? m % n : (n - abs(m % n)) % n; }
__global__ void data_transfer(float2* source, float2* destination, const data_transfer_params consts)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < consts.arrayCount) {
		int d3 = id / (consts.rect_size[0] * consts.rect_size[1]);
		int d12 = id - (d3 * consts.rect_size[0] * consts.rect_size[1]);
		int d2 = d12 / consts.rect_size[0];
		int d1 = d12 - (consts.rect_size[0] * d2);

		destination[(d1 + consts.rect_destination[0]) + (d2 + consts.rect_destination[1]) * consts.size_destination[0] + (d3 + consts.rect_destination[2]) * consts.size_destination[0] * consts.size_destination[1]] =
			source[(d1 + consts.rect_source[0]) + (d2 + consts.rect_source[1]) * consts.size_source[0] + (d3 + consts.rect_source[2]) * consts.size_source[0] * consts.size_source[1]];
	}
}
void parameter_born::transfer_data(std::shared_ptr<CUDA_ARRAY<float2>> source, std::array<std::array<size_t, 2>, 3> source_rectangle, std::shared_ptr<CUDA_ARRAY<float2>> destination, std::array<std::array<size_t, 2>, 3> destination_rectangle) {
	transfer_data(source, source_rectangle, destination, destination_rectangle, source->get_size(), destination->get_size());
}
void parameter_born::transfer_data(std::shared_ptr<CUDA_ARRAY<float2>> source, std::array<std::array<size_t,2>,3> source_rectangle, std::shared_ptr<CUDA_ARRAY<float2>> destination, std::array<std::array<size_t, 2>,3> destination_rectangle, std::vector<size_t> sz_source, std::vector<size_t> sz_destination) {
	//swap the size if not in the good order
	if (source->get_size().size()<3) {
		throw(std::runtime_error("parameter_born::transfer_data::Please upload your source array as 3D array ('you might set some dimention to 1')"));
		return;
	}
	if (destination->get_size().size() < 3) {
		throw(std::runtime_error("parameter_born::transfer_data::Please upload your destination array as 3D array ('you might set some dimention to 1')"));
		return;
	}
	for(int i = 0; i < source_rectangle.size(); i++) {
		if (source_rectangle[i][0] > source_rectangle[i][1]) {
			auto temp = source_rectangle[i][0];
			source_rectangle[i][0] = source_rectangle[i][1];
			source_rectangle[i][1] = temp;
		}
	}
	for(int i = 0; i < destination_rectangle.size(); i++) {
		if (destination_rectangle[i][0] > destination_rectangle[i][1]) {
			auto temp = destination_rectangle[i][0];
			source_rectangle[i][0] = destination_rectangle[i][1];
			source_rectangle[i][1] = temp;
		}
	}
	//check that both have the same size
	int arrayCount = 1;
	for(int i = 0; i < destination_rectangle.size(); i++) {
		arrayCount = arrayCount* (destination_rectangle[i][1] - destination_rectangle[i][0]+1);
		if (destination_rectangle[i][1] - destination_rectangle[i][0] + 1 != source_rectangle[i][1] - source_rectangle[i][0] + 1) {
			throw(std::runtime_error("parameter_born::transfer_data::The data transfer input and output rectangle do not match"));
			return;
		}
	}
	for (int i = 0; i < destination_rectangle.size(); i++) {
		if (destination_rectangle[i][0]<0||destination_rectangle[i][1]>= sz_destination[i]|| source_rectangle[i][0] < 0 || source_rectangle[i][1] >= sz_source[i]) {
			throw(std::runtime_error("parameter_born::transfer_data::rectangle is out of bound along dim " + std::to_string(i+1) + " with "
				+ std::to_string(destination_rectangle[i][0]) + ":" + std::to_string(destination_rectangle[i][1]) + "/"
				+ std::to_string(source_rectangle[i][0]) + ":" + std::to_string(source_rectangle[i][1]) + "/"
				+ std::to_string(sz_destination[i]) + "/" + std::to_string(sz_source[i])
			));
			return;
		}
	}
	data_transfer_params params;
	params.arrayCount = arrayCount;
	for (int i = 0; i < destination_rectangle.size(); i++) {
		params.rect_source[i] = source_rectangle[i][0];
		params.rect_destination[i] = destination_rectangle[i][0];
		params.rect_size[i] = source_rectangle[i][1] - source_rectangle[i][0] + 1;
		params.size_source[i] = sz_source[i];
		params.size_destination[i] = sz_destination[i];
	}

	int blockSize_1;
	int minGridSize_1;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize_1, &blockSize_1, (void*)data_transfer, 0, arrayCount);
	int gridSize_1 = (arrayCount + blockSize_1 - 1) / blockSize_1;

	if (display_debug) {
		mexPrintf("Transfer data :  \n");
		mexPrintf("    :arrayCount:%d  \n", params.arrayCount);
		mexPrintf("    :rect_source:  \n");
		mexPrintf("        :%d  \n", params.rect_source[0]);
		mexPrintf("        :%d  \n", params.rect_source[1]);
		mexPrintf("        :%d  \n", params.rect_source[2]);
		mexPrintf("    :rect_destination:  \n");
		mexPrintf("        :%d  \n", params.rect_destination[0]);
		mexPrintf("        :%d  \n", params.rect_destination[1]);
		mexPrintf("        :%d  \n", params.rect_destination[2]);
		mexPrintf("    :rect_size:  \n");
		mexPrintf("        :%d  \n", params.rect_size[0]);
		mexPrintf("        :%d  \n", params.rect_size[1]);
		mexPrintf("        :%d  \n", params.rect_size[2]);
		mexPrintf("    :size_source:  \n");
		mexPrintf("        :%d  \n", params.size_source[0]);
		mexPrintf("        :%d  \n", params.size_source[1]);
		mexPrintf("        :%d  \n", params.size_source[2]);
		mexPrintf("    :size_destination:  \n");
		mexPrintf("        :%d  \n", params.size_destination[0]);
		mexPrintf("        :%d  \n", params.size_destination[1]);
		mexPrintf("        :%d  \n", params.size_destination[2]);
	}

	data_transfer << <gridSize_1, blockSize_1 >> > (source->cuda_ptr(),destination->cuda_ptr(), params);

}
struct field_params {
	float normalization_fft;
	int size[3];
	int arrayCount;
};
__global__ void field_in_kernel(float2* field, float2* temp_out, float2* green, const field_params consts, const int z_start)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < consts.arrayCount) {
		int d3 = id / (consts.size[0] * consts.size[1]);
		int d12 = id - (d3 * consts.size[0] * consts.size[1]);
		int d2 = d12 / consts.size[0];
		int d1 = d12 - (consts.size[0] * d2);

		d3 = d3 + z_start;

		int d1_shift = consts.size[0] / 2 + (modulo_cpy(float(d1) + floorf(float(consts.size[0]) / 2.), consts.size[0]) - floorf(float(consts.size[0]) / 2.));
		int d2_shift = consts.size[0] / 2 + (modulo_cpy(float(d2) + floorf(float(consts.size[1]) / 2.), consts.size[1]) - floorf(float(consts.size[1]) / 2.));

		if (d3 < consts.size[2]) {
			int green_d1 = (consts.size[0] / 2)-abs(d1_shift - consts.size[0] / 2);
			int green_d2 = (consts.size[1] / 2)-abs(d2_shift - consts.size[1] / 2);
			int green_d3 = (consts.size[2] / 2)-abs(d3 - consts.size[2] / 2);

			bool do_conj = (d3 - consts.size[2] / 2) < 0;

			int green_sz1 = (consts.size[0] / 2)+1;
			int green_sz2 = (consts.size[1] / 2)+1;
			int green_sz3 = (consts.size[2] / 2)+1;

			float2 coeff =  green[green_d1 + green_d2 * green_sz1 + green_d3 * green_sz1 * green_sz2];
			if (!do_conj) coeff=cuConjf(coeff);
			float2 fld = field[d1+d2* consts.size[0]];
			temp_out[id] = cuCmulf(cuCmulf(fld, coeff), make_cuFloatComplex(consts.normalization_fft, 0));
			//temp_out[id] = make_cuFloatComplex(d1_shift, d2_shift);
		}
	}
}
__global__ void field_out_kernel(float2* field, float2* field2, float2* temp_out, float2* green, const field_params consts, const int z_start,const int conv_size_z,const float dV)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < consts.arrayCount) {
		
		int d12 = id;
		int d2 = d12 / consts.size[0];
		int d1 = d12 - (consts.size[0] * d2);


		float2 fld = make_cuFloatComplex(0, 0);
		float2 fld2 = make_cuFloatComplex(0, 0);

		float2 coeff;

		int d1_shift = consts.size[0] / 2 + (modulo_cpy(float(d1) + floorf(float(consts.size[0]) / 2.), consts.size[0]) - floorf(float(consts.size[0]) / 2.));
		int d2_shift = consts.size[0] / 2 + (modulo_cpy(float(d2) + floorf(float(consts.size[1]) / 2.), consts.size[1]) - floorf(float(consts.size[1]) / 2.));
		int green_d1 = (consts.size[0] / 2) - abs(d1_shift - consts.size[0] / 2);
		int green_d2 = (consts.size[1] / 2) - abs(d2_shift - consts.size[1] / 2);

		int green_sz1 = (consts.size[0] / 2) + 1;
		int green_sz2 = (consts.size[1] / 2) + 1;
		int green_sz3 = (consts.size[2] / 2) + 1;

		//temp_out[id] = make_cuFloatComplex(3, 3);

		for (int i = 0; i < conv_size_z; i++) {
			int d3 = i + z_start;

			if (d3 < consts.size[2]) {
				
				int green_d3 = (consts.size[2] / 2) - abs(d3 - consts.size[2] / 2);
				bool do_conj = (d3 - consts.size[2] / 2) < 0;

				coeff = green[green_d1 + green_d2 * green_sz1 + green_d3 * green_sz1 * green_sz2];
				if (!do_conj) coeff = cuCsubf(make_cuFloatComplex(0.,0.),cuConjf(coeff));

				
				float2 curr_field = temp_out[d1 + d2 * consts.size[0] + i * (consts.size[0]*consts.size[1])];
				fld = cuCaddf(fld,cuCmulf(cuCmulf(curr_field, (coeff)), make_cuFloatComplex(consts.normalization_fft*dV, 0)));
				fld2 = cuCaddf(fld2, cuCmulf(cuCmulf(curr_field, cuConjf(coeff)), make_cuFloatComplex(consts.normalization_fft * dV, 0)));
				//temp_out[d1 + d2 * consts.size[0] + i * (consts.size[0] * consts.size[1])] = coeff;// (cuCmulf(curr_field, (coeff)));
			}
		}
		field[id] = cuCaddf(field[id], fld);
		field2[id] = cuCaddf(field2[id], fld2);
	}
}

void parameter_born::execute_2D_in() {
	
	std::array<std::array<size_t, 2>, 3> rect_src_field2conv = { {{0,field_size[0] - 1},{0,field_size[1] - 1},{0,0}} };
	std::array<std::array<size_t, 2>, 3> rect_dst_field2conv = { { {(int)conv_size[0] / 2 - field_size[0] / 2,0},{(int)conv_size[1] / 2 - field_size[1] / 2,0},{0,0} } };
	rect_dst_field2conv[0][1] = rect_dst_field2conv[0][0] + field_size[0] - 1;
	rect_dst_field2conv[1][1] = rect_dst_field2conv[1][0] + field_size[1] - 1;

	std::array<std::array<size_t, 2>, 3> rect_src_conv2ri = { {{(int)conv_size[0] / 2 - RI_size[0] / 2,0},{(int)conv_size[1] / 2 - RI_size[1] / 2,0},{0,0}} };
	rect_src_conv2ri[0][1] = rect_src_conv2ri[0][0] + RI_size[0] - 1;
	rect_src_conv2ri[1][1] = rect_src_conv2ri[1][0] + RI_size[1] - 1;
	std::array<std::array<size_t, 2>, 3> rect_dst_conv2ri = { {{(int)simulation_size[0] / 2 - RI_size[0] / 2,0},{(int)simulation_size[1] / 2 - RI_size[1] / 2,0},{0,0}} };
	rect_dst_conv2ri[0][1] = rect_dst_conv2ri[0][0] + RI_size[0] - 1;
	rect_dst_conv2ri[1][1] = rect_dst_conv2ri[1][0] + RI_size[1] - 1;
	auto border_z_conv2ri = (int)simulation_size[2] / 2 - RI_size[2] / 2;

	std::array<std::array<size_t, 2>, 3> rect_conv2conv = { {{0,conv_size[0] - 1},{0,conv_size[1] - 1},{0,conv_size[2] - 1}} };
	std::array<std::array<size_t, 2>, 3> rect_conv2conv_small = { {{0,conv_size[0] - 1},{0,conv_size[1] - 1},{0,0}} };

	int arrayCount = conv_size[0] * conv_size[1] * conv_size[2];
	int blockSize_1;
	int minGridSize_1;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize_1, &blockSize_1, (void*)field_in_kernel, 0, arrayCount);
	int gridSize_1 = (arrayCount + blockSize_1 - 1) / blockSize_1;

	field_params consts;
	consts.normalization_fft = 1. / float(conv_size[0] * conv_size[1]);
	consts.size[0] = conv_size[0];
	consts.size[1] = conv_size[1];
	consts.size[2] = RI_size[2];
	consts.arrayCount = arrayCount;

	

	if (display_debug)mexPrintf("Transfer source \n");
	temp_fft2->zero_all();
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		curr_gpu=loaded_gpu[gpu];
		transfer_data(source_2D, rect_src_field2conv, temp_fft2, rect_dst_field2conv);
	}
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		fft2(temp_fft2,loaded_gpu[gpu]);
	}
	if (display_debug)mexPrintf("Transfer temp \n");
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		curr_gpu=loaded_gpu[gpu];
		transfer_data(temp_fft2, rect_conv2conv_small, temp_field, rect_conv2conv_small);
	}
	if (display_debug)mexPrintf("Execute \n");
	
	for (int i = 0; i < ((RI_size[2]-1)/conv_size[2])+1; i++) {
		//cudaDeviceSynchronize();
		for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
			cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
			field_in_kernel << <gridSize_1, blockSize_1 >> > (temp_field->cuda_ptr(loaded_gpu[gpu]), temp_fft2->cuda_ptr(loaded_gpu[gpu]), free_space_green->cuda_ptr(loaded_gpu[gpu]), consts, i * conv_size[2]);
		}
		for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
			ifft2(temp_fft2,loaded_gpu[gpu]);
		}

		rect_dst_conv2ri[2][0] = i* conv_size[2];
		rect_dst_conv2ri[2][1] = min((i + 1) * conv_size[2] - 1, RI_size[2]);
		rect_dst_conv2ri[2][0] += border_z_conv2ri;
		rect_dst_conv2ri[2][1] += border_z_conv2ri;

		rect_src_conv2ri[2][1] = rect_dst_conv2ri[2][1] - rect_dst_conv2ri[2][0];
		if (display_debug)mexPrintf("Write result \n");
		for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
			cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
			curr_gpu=loaded_gpu[gpu];
			transfer_data(temp_fft2, rect_src_conv2ri, var1, rect_dst_conv2ri);
		}
	}
	if (display_debug)mexPrintf("Finished the input\n");
}

void parameter_born::execute_2D_out() {

	int arrayCount = conv_size[0] * conv_size[1] ;
	int blockSize_1;
	int minGridSize_1;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize_1, &blockSize_1, (void*)field_out_kernel, 0, arrayCount);
	int gridSize_1 = (arrayCount + blockSize_1 - 1) / blockSize_1;

	field_params consts;
	consts.normalization_fft = 1. / float(conv_size[0] * conv_size[1]);
	consts.size[0] = conv_size[0];
	consts.size[1] = conv_size[1];
	consts.size[2] = RI_size[2];
	consts.arrayCount = arrayCount;


	std::array<std::array<size_t, 2>, 3> rect_src_sim2ri = { {{(int)simulation_size[0] / 2 - RI_size[0] / 2,0},{(int)simulation_size[1] / 2 - RI_size[1] / 2,0},{(int)simulation_size[2] / 2 - RI_size[2] / 2,0}} };
	rect_src_sim2ri[0][1] = rect_src_sim2ri[0][0] + RI_size[0] - 1;
	rect_src_sim2ri[1][1] = rect_src_sim2ri[1][0] + RI_size[1] - 1;
	rect_src_sim2ri[2][1] = rect_src_sim2ri[2][0] + RI_size[2] - 1;
	std::array<std::array<size_t, 2>, 3> rect_src_ri2ri = { {{0, RI_size[0] - 1},{0, RI_size[1] - 1},{0, RI_size[2] - 1}} };

	std::array<std::array<size_t, 2>, 3> rect_conv2conv_1 = { {{0,conv_size[0] - 1},{0,conv_size[1] - 1},{0,0}} };
	std::array<std::array<size_t, 2>, 3> rect_conv2conv_2 = { {{0,conv_size[0] - 1},{0,conv_size[1] - 1},{1,1}} };

	std::array<std::array<size_t, 2>, 3> rect_dst_field2conv = { {{0,field_size[0] - 1},{0,field_size[1] - 1},{0,0}} };
	std::array<std::array<size_t, 2>, 3> rect_src_field2conv = { { {(int)conv_size[0] / 2 - field_size[0] / 2,0},{(int)conv_size[1] / 2 - field_size[1] / 2,0},{0,0} } };
	rect_src_field2conv[0][1] = rect_src_field2conv[0][0] + field_size[0] - 1;
	rect_src_field2conv[1][1] = rect_src_field2conv[1][0] + field_size[1] - 1;

	std::vector<size_t> RI_sz(RI_size.begin(), RI_size.end());
	std::vector<size_t> SI_sz(simulation_size.begin(), simulation_size.end());
	//3D field crop
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		curr_gpu=loaded_gpu[gpu];
		transfer_data(var3, rect_src_sim2ri, var1, rect_src_ri2ri, SI_sz, RI_sz);
	}
	//3D emission crop
	//transfer_data(var2, rect_src_sim2ri, var3, rect_src_ri2ri, SI_sz, RI_sz);
	//2D fields compute
	std::array<std::array<size_t, 2>, 3> rect_dst_conv2ri = { {{(int)conv_size[0] / 2 - RI_size[0] / 2,0},{(int)conv_size[1] / 2 - RI_size[1] / 2,0},{0,0}} };
	rect_dst_conv2ri[0][1] = rect_dst_conv2ri[0][0] + RI_size[0] - 1;
	rect_dst_conv2ri[1][1] = rect_dst_conv2ri[1][0] + RI_size[1] - 1;
	std::array<std::array<size_t, 2>, 3> rect_src_conv2ri = { {{(int)simulation_size[0] / 2 - RI_size[0] / 2,0},{(int)simulation_size[1] / 2 - RI_size[1] / 2,0},{0,0}} };
	rect_src_conv2ri[0][1] = rect_src_conv2ri[0][0] + RI_size[0] - 1;
	rect_src_conv2ri[1][1] = rect_src_conv2ri[1][0] + RI_size[1] - 1;

	temp_field->zero_all();
	temp_field_2->zero_all();
	temp_fft2->zero_all();
	auto border_z_conv2ri = (int)simulation_size[2] / 2 - RI_size[2] / 2;

	for (int i = 0; i < ((RI_size[2] - 1) / conv_size[2]) + 1; i++) {

		rect_src_conv2ri[2][0] = i * conv_size[2];
		rect_src_conv2ri[2][1] = min((i + 1) * conv_size[2] - 1, RI_size[2]);
		rect_src_conv2ri[2][0] += border_z_conv2ri;
		rect_src_conv2ri[2][1] += border_z_conv2ri;

		rect_dst_conv2ri[2][1] = rect_src_conv2ri[2][1] - rect_src_conv2ri[2][0];

		temp_fft2->zero_all();
		//temp_field->zero();
		//temp_field_2->zero();
		for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
			cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
			curr_gpu=loaded_gpu[gpu];
			transfer_data(var2, rect_src_conv2ri, temp_fft2, rect_dst_conv2ri);
		}
		for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
			fft2(temp_fft2,loaded_gpu[gpu]);
		}

		//mexPrintf(" var : %d ; %d ; %d ; %d ; %d ;%d;%d;%d \n", conv_size[2], i * conv_size[2], conv_size[0] ,conv_size[1], RI_size[2], gridSize_1, blockSize_1, consts.arrayCount);
		for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
			cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
			field_out_kernel << <gridSize_1, blockSize_1 >> > (temp_field->cuda_ptr(loaded_gpu[gpu]), temp_field_2->cuda_ptr(loaded_gpu[gpu]), temp_fft2->cuda_ptr(loaded_gpu[gpu]), free_space_green_2->cuda_ptr(loaded_gpu[gpu]), consts, i * conv_size[2], conv_size[2], dV);
		}
		//mexPrintf(" done \n");
	}
	
	//mexPrintf(" 1 \n");
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		curr_gpu=loaded_gpu[gpu];
		transfer_data(temp_field, rect_conv2conv_1, temp_fft2, rect_conv2conv_1);
	}
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		curr_gpu=loaded_gpu[gpu];
		transfer_data(temp_field_2, rect_conv2conv_1, temp_fft2, rect_conv2conv_2);
	}
	//mexPrintf(" 2 \n");
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		ifft2(temp_fft2,loaded_gpu[gpu]);
	}
	rect_src_field2conv[2][0] = 0; rect_src_field2conv[2][1] = 0;
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		curr_gpu=loaded_gpu[gpu];
		transfer_data(temp_fft2, rect_src_field2conv, trans_2D, rect_dst_field2conv);
	}
	//mexPrintf(" 3 \n");
	rect_src_field2conv[2][0] = 1; rect_src_field2conv[2][1] = 1;
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		curr_gpu=loaded_gpu[gpu];
		transfer_data(temp_fft2, rect_src_field2conv, ref_2D, rect_dst_field2conv);
	}
	//mexPrintf(" 4 \n");
	
}


