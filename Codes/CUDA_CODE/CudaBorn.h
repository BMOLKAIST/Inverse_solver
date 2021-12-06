#ifndef CUDABORN_H
#define CUDABORN_H

#include"static_parameter.h"
#include"memory_management_cuda.h"
#include <array>
#include <cufft.h>
#include <memory>

class parameter_born {
private:
	bool output_ready = false;
public:
	//forbid copy
	parameter_born(const parameter_born&) = delete;
	void operator=(const parameter_born&) = delete;
	//class construct/destruct
	parameter_born();
	~parameter_born();
	

	//CPU data
	float eps_imag;
	float k0_nm;
	float dV;
	std::array<float,3> fourier_res;
	std::array<uint32_t, 3> simulation_size;
	std::array<uint32_t, 3> RI_size;
	std::array<uint32_t, 3> field_size;
	std::array<uint32_t, 3> conv_size;
	std::array<uint32_t, 3> green_size;
	std::array<uint32_t, 6> ROI_simulation;
    std::vector<uint32_t> used_gpu;
	std::vector<size_t> loaded_gpu;
	
    
    uint32_t curr_gpu=0;
	uint32_t Bornmax;
    
    
	//GPU data
	std::shared_ptr<CUDA_ARRAY<float2>> var1;
	std::shared_ptr<CUDA_ARRAY<float2>> var2;
	std::shared_ptr<CUDA_ARRAY<float2>> var3;

	std::shared_ptr<CUDA_ARRAY<float2>> free_space_green;
	std::shared_ptr<CUDA_ARRAY<float2>> free_space_green_2;
	std::shared_ptr<CUDA_ARRAY<float2>> temp_fft2;

	std::shared_ptr<CUDA_ARRAY<float2>> source_2D;
	std::shared_ptr<CUDA_ARRAY<float2>> trans_2D;
	std::shared_ptr<CUDA_ARRAY<float2>> ref_2D;

	std::shared_ptr<CUDA_ARRAY<float2>> temp_field;
	std::shared_ptr<CUDA_ARRAY<float2>> temp_field_2;

	std::shared_ptr<CUDA_ARRAY<float2>> scattering_potential;
	//fft plans
	std::vector<int> fft_size;
	std::vector<cufftHandle> fft_plan;
	bool fft_ready = false;

	void prepare_fft3();
	void destroy_fft3();
	void fft3(std::shared_ptr<CUDA_ARRAY<float2>>);
	void fft3(std::shared_ptr<CUDA_ARRAY<float2>>, size_t gpu);
	void ifft3(std::shared_ptr<CUDA_ARRAY<float2>>);
	void ifft3(std::shared_ptr<CUDA_ARRAY<float2>>, size_t gpu);


	std::vector<int> fft_2D_size;
	std::vector<cufftHandle> fft_2D_plan;
	bool fft_2D_ready = false;

	void prepare_fft2();
	void destroy_fft2();
	void fft2(std::shared_ptr<CUDA_ARRAY<float2>>);
	void fft2(std::shared_ptr<CUDA_ARRAY<float2>>, size_t gpu);
	void ifft2(std::shared_ptr<CUDA_ARRAY<float2>>);
	void ifft2(std::shared_ptr<CUDA_ARRAY<float2>>, size_t gpu);
	//execution
	void execute();
	void execute_2D_in();
	void execute_2D_out();
	//data transfer
    void check_GPU();
	void transfer_data(std::shared_ptr<CUDA_ARRAY<float2>> source, std::array<std::array<size_t, 2>, 3> source_rectangle, std::shared_ptr<CUDA_ARRAY<float2>> destination, std::array<std::array<size_t, 2>, 3> destination_rectangle);
	void transfer_data(std::shared_ptr<CUDA_ARRAY<float2>> source, std::array<std::array<size_t, 2>, 3> source_rectangle, std::shared_ptr<CUDA_ARRAY<float2>> destination, std::array<std::array<size_t, 2>, 3> destination_rectangle, std::vector<size_t> sz_source, std::vector<size_t> sz_destination);
	//data function
	void set_V(mxArray const* const matlab_data);
	void set_source_3D(mxArray const* const data);
	void set_source_2D(mxArray const* const data);
	void set_green(mxArray const* const data);
	void set_green_2(mxArray const* const data);
	void get_field_3D(mxArray*& data);
	void get_trans_2D(mxArray*& data);
	void get_ref_2D(mxArray*& data);
	//execution function
	void create_cuda_data();
	void delete_cuda_data();
	//display function
	void display_state();

};



#endif