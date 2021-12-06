#include"static_parameter.h"
#include"CudaBorn.h"
#include"memory_management_cuda.h"
#include <cufft.h>
#include <mex.h>
#include "cuComplex.h"
#include <complex.h>
#include "math_constants.h"
#include <algorithm>

void parameter_born::prepare_fft3() {
	int curr_gpu;
	cudaGetDevice(&curr_gpu);
	destroy_fft3();
	fft_size = std::vector<int>(simulation_size.begin(), simulation_size.end());
	auto cufft_adapt_size = fft_size;
	std::reverse(cufft_adapt_size.begin(), cufft_adapt_size.end());
	int batch_number = 1;
	int linear_size_3D=std::accumulate(begin(fft_size), end(fft_size), 1, std::multiplies<int>());
	/* Create a 3D FFT plan. */
	fft_plan=std::vector<cufftHandle>(used_gpu.size());
	for (int gpu=0; gpu<used_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[gpu]);
		if (cufftPlanMany(&(fft_plan[gpu]), cufft_adapt_size.size(), cufft_adapt_size.data(),
			NULL, 1, linear_size_3D, // *inembed, istride, idist 
			NULL, 1, linear_size_3D, // *onembed, ostride, odist
			CUFFT_C2C, batch_number) != CUFFT_SUCCESS) {
			throw(std::runtime_error("parameter_born::prepare_fft:: CUFFT error: Plan creation failed"));
			return;
		}
	}
	cudaSetDevice (curr_gpu);
	fft_ready = true;
}

void parameter_born::destroy_fft3() {
	int curr_gpu;
	cudaGetDevice(&curr_gpu);
	for (int gpu=0; gpu<used_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[gpu]);
		if(fft_ready) {
			cufftDestroy(fft_plan[gpu]);
			if (display_debug)mexPrintf("fftn deleted\n");
		}else{
			if (display_debug)mexPrintf("fftn not deleted\n");
		}
	}
	cudaSetDevice (curr_gpu);
	fft_ready = false;
}
void parameter_born::fft3(std::shared_ptr<CUDA_ARRAY<float2>> var) {
	fft3(var,curr_gpu);
}
void parameter_born::fft3(std::shared_ptr<CUDA_ARRAY<float2>> var, size_t gpu) {
	cudaSetDevice(used_gpu[gpu]);
	if(!fft_ready) throw(std::runtime_error("parameter_born::fft3::fft is not ready"));
	std::vector<size_t> fft_size_size_t(fft_size.begin(), fft_size.end());
	if (fft_size_size_t!=var->get_size()) throw(std::runtime_error("parameter_born::fft3::size not matching"));
	if (cufftExecC2C(fft_plan[gpu], var->cuda_ptr(gpu), var->cuda_ptr(gpu), CUFFT_FORWARD) != CUFFT_SUCCESS) {
		throw(std::runtime_error("parameter_born::fft3::CUFFT error: ExecC2C Forward failed"));
		return;
	}
}
void parameter_born::ifft3(std::shared_ptr<CUDA_ARRAY<float2>> var) {
	ifft3(var,curr_gpu);
}
void parameter_born::ifft3(std::shared_ptr<CUDA_ARRAY<float2>> var, size_t gpu) {
	cudaSetDevice(used_gpu[gpu]);
	if (!fft_ready) throw(std::runtime_error("parameter_born::ifft3::ifft is not ready"));
	std::vector<size_t> fft_size_size_t(fft_size.begin(), fft_size.end());
	if (fft_size_size_t != var->get_size()) throw(std::runtime_error("parameter_born::fft3::size not matching"));
	if (cufftExecC2C(fft_plan[gpu], var->cuda_ptr(gpu), var->cuda_ptr(gpu), CUFFT_INVERSE) != CUFFT_SUCCESS) {
		throw(std::runtime_error("parameter_born::fft3::CUFFT error: ExecC2C Forward failed"));
		return;
	}
}

struct fourier_params {
	int arrayCount;
	int size[3];
	float normalisation_fft;
	float eps_imag;
	float k0_nm;
	float fourier_res[3];
};
__device__ int modulo(int m, int n) { return m >= 0 ? m % n : (n - abs(m % n)) % n; }
__global__ void fourier_kernel(float2* matt, const fourier_params consts, const bool direction)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < consts.arrayCount) {
		int d3 = id / (consts.size[0] * consts.size[1]);
		int d12 = id - (d3 * consts.size[0] * consts.size[1]);
		int d2 = d12 / consts.size[0];
		int d1 = d12 - (d2 * consts.size[0]);

		float shift = direction ? 1./4. : -1./4.;
		//shift = 0;

		float k1 = consts.fourier_res[0] * (shift + (modulo(float(d1) + floorf(float(consts.size[0]) / 2.), consts.size[0]) - floorf(float(consts.size[0]) / 2.)));
		float k2 = consts.fourier_res[1] * (shift + (modulo(float(d2) + floorf(float(consts.size[1]) / 2.), consts.size[1]) - floorf(float(consts.size[1]) / 2.)));
		float k3 = consts.fourier_res[2] * (shift + (modulo(float(d3) + floorf(float(consts.size[2]) / 2.), consts.size[2]) - floorf(float(consts.size[2]) / 2.)));

		float2 matt_temp = matt[id];

		matt_temp = cuCmulf(make_cuFloatComplex(consts.normalisation_fft, 0), matt_temp);

		//matt_temp = make_cuFloatComplex(1, 0);

		matt_temp = cuCdivf(matt_temp, make_cuFloatComplex(4 * CUDART_PI_F * CUDART_PI_F * (k1 * k1 + k2 * k2 + k3 * k3 - consts.k0_nm * consts.k0_nm), -consts.eps_imag));

		matt[id] = matt_temp;
	}
}
struct imag_params {
	int arrayCount;
	int size[3];
	float eps_imag;
	int ROI[6];
};
__device__ void get_ramp(int id, const imag_params consts, const bool direction, float2 & ramp, float & mask) {
	
	int d3 = id / (consts.size[0] * consts.size[1]);
	int d12 = id - (d3 * consts.size[0] * consts.size[1]);
	int d2 = d12 / consts.size[0];
	int d1 = d12 - (consts.size[0] * d2);
	//ramp
	float ramp_angle =
		(float(d1) - floorf(float(consts.size[0]) / 2.)) / float(consts.size[0] * 2) +
		(float(d2) - floorf(float(consts.size[1]) / 2.)) / float(consts.size[1] * 2) +
		(float(d3) - floorf(float(consts.size[2]) / 2.)) / float(consts.size[2] * 2);
	float sign_ramp = direction ? 1. : -1.;
	sincosf(sign_ramp * CUDART_PI_F * ramp_angle, &ramp.y, &ramp.x);
	//ramp = make_cuFloatComplex(1, 0);
	//mask
	float dist1 = fmaxf(fmaxf((float(consts.ROI[0] - d1) - 1.5) / (float(consts.ROI[0]) - 1.5), 0), fmaxf((float(d1 - (consts.ROI[1])) + 0.5) / (float(consts.size[0] - consts.ROI[1]) - 0.5), 0));
	float dist2 = fmaxf(fmaxf((float(consts.ROI[2] - d2) - 1.5) / (float(consts.ROI[2]) - 1.5), 0), fmaxf((float(d2 - (consts.ROI[3])) + 0.5) / (float(consts.size[1] - consts.ROI[3]) - 0.5), 0));
	float dist3 = fmaxf(fmaxf((float(consts.ROI[4] - d3) - 1.5) / (float(consts.ROI[4]) - 1.5), 0), fmaxf((float(d3 - (consts.ROI[5])) + 0.5) / (float(consts.size[2] - consts.ROI[5]) - 0.5), 0));
	mask=(1-dist1)*(1-dist2)*(1-dist3);
}
__global__ void imag_kernel(float2* field, float2* field_n, float2* fourier_array, float2 const * const scattering_potential, const imag_params consts, const bool direction)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < consts.arrayCount) {
		float2 ramp;
		float mask;
		get_ramp(id, consts, direction, ramp, mask);


		float2 f_n = field_n[id];
		float2 f = field[id];
		float2 V = scattering_potential[id];
		float2 fourier = fourier_array[id];
		//second part
		fourier = cuCmulf(fourier, (ramp));
		f_n = cuCaddf(f_n, cuCmulf(V, fourier));
		f_n = cuCmulf(f_n, make_cuFloatComplex(mask, 0));
		f = cuCaddf(f, f_n);
		//first part
		fourier = cuCmulf(cuCmulf(f_n, make_cuFloatComplex(0, 1. / consts.eps_imag)), V);
		f_n = cuCsubf(f_n, fourier);
		fourier = cuCmulf(fourier, (ramp));

		fourier_array[id] = fourier;
		field_n[id] = f_n;
		field[id] = f;

		//field[id] = make_cuFloatComplex(mask,0);
	}
}
__global__ void init1_kernel(float2* field, float2* field_n, float2* fourier_array, float2 const* const scattering_potential, const imag_params consts, const bool direction)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < consts.arrayCount) {
		float2 ramp;
		float mask;
		get_ramp(id, consts, direction, ramp, mask);

		float2 source = fourier_array[id];
		source = cuCmulf(source, cuCaddf(scattering_potential[id],make_cuFloatComplex(0, consts.eps_imag)));
		source = cuCmulf(source, make_cuFloatComplex(0, 1. / (2. * consts.eps_imag)));
		field_n[id] = cuCmulf(source, cuConjf(ramp));
		field[id] = cuCmulf(source, (ramp));
	}
}
__global__ void init2_kernel(float2* field, float2* field_n, float2* fourier_array, float2 const* const scattering_potential, const imag_params consts, const bool direction)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < consts.arrayCount) {
		float2 ramp;
		float mask;
		get_ramp(id, consts, direction, ramp, mask);

		float2 incident_field = fourier_array[id];

		float2 V = scattering_potential[id];
		float2 fourier1 = cuCmulf(field_n[id], (ramp));
		float2 fourier2 = cuCmulf(field[id], cuConjf(ramp));

		fourier1 = cuCmulf(fourier1, V);
		fourier2 = cuCmulf(fourier2, V);

		float2 fourier1_before_mask = fourier1;
		fourier1 = cuCmulf(fourier1, make_cuFloatComplex(mask, 0));
		fourier2 = cuCmulf(fourier2, make_cuFloatComplex(mask, 0));

		field[id] = cuCaddf(fourier2,incident_field);

		float2 f_n = fourier2;
		float2 fourier = cuCmulf(cuCmulf(f_n, make_cuFloatComplex(0, 1. / consts.eps_imag)), V);
		f_n = cuCsubf(f_n, fourier);
		f_n = cuCaddf(f_n, fourier1_before_mask);
		fourier = cuCmulf(fourier, ramp);

		fourier_array[id] = fourier;
		field_n[id] = f_n;
	}
}
__global__ void out_prep_kernel(float2* field, float2* field_n, float2* fourier_array, float2 const* const scattering_potential, const imag_params consts, const bool direction)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < consts.arrayCount) {
		float2 fld = field[id];
		fld = cuCmulf(fld, cuCaddf(scattering_potential[id], make_cuFloatComplex(0, consts.eps_imag)));
		field_n[id] = fld;
	}
}
void parameter_born::execute() {

	if (display_debug)mexPrintf("1 \n");
	bool direction = false;

	if (!fft_ready) throw(std::runtime_error("parameter_born::execute::fft is not ready"));
	int arrayCount = std::accumulate(begin(fft_size), end(fft_size), 1, std::multiplies<int>());
	if (display_debug)mexPrintf("1.1 \n");
	//mexPrintf(" count : %f\n",  float(arrayCount));
	//set constants
	const fourier_params fourier_const = { arrayCount, {fft_size[0],fft_size[1],fft_size[2]},1./float(arrayCount),eps_imag,k0_nm,{fourier_res[0],fourier_res[1],fourier_res[2]} };
	const imag_params	 imag_const =    { arrayCount, {fft_size[0],fft_size[1],fft_size[2]},eps_imag,{ROI_simulation[0],ROI_simulation[1],ROI_simulation[2],ROI_simulation[3],ROI_simulation[4],ROI_simulation[5]} };
	//prepare kernels
	if (display_debug)mexPrintf("1.2 \n");
	int blockSize_1;
	int minGridSize_1;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize_1, &blockSize_1, (void*)fourier_kernel, 0, arrayCount);
	int gridSize_1 = (arrayCount + blockSize_1 - 1) / blockSize_1;
	
	int blockSize_2;
	int minGridSize_2;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize_2, &blockSize_2, (void*)imag_kernel, 0, arrayCount);
	int gridSize_2 = (arrayCount + blockSize_2 - 1) / blockSize_2;
	if (display_debug)mexPrintf("1.3 \n");
	int blockSize_3;
	int minGridSize_3;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize_3, &blockSize_3, (void*)init1_kernel, 0, arrayCount);
	int gridSize_3 = (arrayCount + blockSize_3 - 1) / blockSize_3;

	int blockSize_4;
	int minGridSize_4;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize_4, &blockSize_4, (void*)init2_kernel, 0, arrayCount);
	int gridSize_4 = (arrayCount + blockSize_4 - 1) / blockSize_4;

	int blockSize_5;
	int minGridSize_5;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize_5, &blockSize_5, (void*)out_prep_kernel, 0, arrayCount);
	int gridSize_5 = (arrayCount + blockSize_5 - 1) / blockSize_5;
	
	//execute
	if (display_debug)mexPrintf("2 \n");
	//elementwise
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		init1_kernel << <gridSize_3, blockSize_3 >> > (var3->cuda_ptr(loaded_gpu[gpu]), var2->cuda_ptr(loaded_gpu[gpu]), var1->cuda_ptr(loaded_gpu[gpu]), scattering_potential->cuda_ptr(loaded_gpu[gpu]), imag_const, direction);
	}
	//fft
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		fft3(var2,loaded_gpu[gpu]);
	}
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		fft3(var3,loaded_gpu[gpu]);
	}

	//elementwise
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		fourier_kernel << <gridSize_1, blockSize_1 >> > (var2->cuda_ptr(loaded_gpu[gpu]), fourier_const, direction);
	}
	direction = !direction;
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		fourier_kernel << <gridSize_1, blockSize_1 >> > (var3->cuda_ptr(loaded_gpu[gpu]), fourier_const, direction);
	}
	direction = !direction;
	
	//fft
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		ifft3(var2,loaded_gpu[gpu]);
	}
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		ifft3(var3,loaded_gpu[gpu]);
	}
	
	//elementwise
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		init2_kernel << <gridSize_4, blockSize_4 >> > (var3->cuda_ptr(loaded_gpu[gpu]), var2->cuda_ptr(loaded_gpu[gpu]), var1->cuda_ptr(loaded_gpu[gpu]), scattering_potential->cuda_ptr(loaded_gpu[gpu]), imag_const, direction);
	}
	
	for (int i = 2; i < Bornmax; i++) {
		//elementwise
		direction = !direction;
		//fft
		for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
			fft3(var1,loaded_gpu[gpu]);
		}
		//elementwise
		for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
			cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
			fourier_kernel << <gridSize_1, blockSize_1 >> > (var1->cuda_ptr(loaded_gpu[gpu]),fourier_const,direction);
		}
		//fft
		for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
			ifft3(var1,loaded_gpu[gpu]);
		}
		//elementwise
		for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
			cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
			imag_kernel << <gridSize_2, blockSize_2 >> > (var3->cuda_ptr(loaded_gpu[gpu]), var2->cuda_ptr(loaded_gpu[gpu]), var1->cuda_ptr(loaded_gpu[gpu]), scattering_potential->cuda_ptr(loaded_gpu[gpu]), imag_const, direction);
		}
		//other
		
		//mexPrintf("direction:%i\n",(int)direction);
	}
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		cudaSetDevice(used_gpu[loaded_gpu[gpu]]);
		out_prep_kernel << <gridSize_5, blockSize_5 >> > (var3->cuda_ptr(loaded_gpu[gpu]), var2->cuda_ptr(loaded_gpu[gpu]), var1->cuda_ptr(loaded_gpu[gpu]), scattering_potential->cuda_ptr(loaded_gpu[gpu]), imag_const, direction);
	}
	output_ready = true;

	//direction = !direction; fourier_kernel << <gridSize_1, blockSize_1 >> > (var1->cuda_ptr(), fourier_const, direction);
}