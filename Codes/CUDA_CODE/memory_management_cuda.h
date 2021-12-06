#ifndef CUDAMEMORY_H
#define CUDAMEMORY_H

#include "static_parameter.h"
#include <vector>
#include <mex.h>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <new> 
#include "cuComplex.h"
#include <cuda_runtime_api.h>
#include <string>

template <class T> class CUDA_ARRAY {
private:
	std::vector<T*> cuda_data;
	size_t data_size_1D;
	std::vector<size_t> data_size;
	std::vector<uint32_t> gpus;
	uint32_t* current_gpu;
public:
	//fobidden copy
	CUDA_ARRAY(const CUDA_ARRAY&) = delete;
	void operator=(const CUDA_ARRAY&) = delete;
	//construction/destruction
	CUDA_ARRAY(size_t size,std::vector<uint32_t> used_gpu,uint32_t* curr_gpu);
	CUDA_ARRAY(std::vector<size_t> size,std::vector<uint32_t> used_gpu,uint32_t* curr_gpu);
	~CUDA_ARRAY();
	//set/get data
	void set_data(T const* const, size_t size=0);//if 0 treate the array as a full array(initial size) else can be used as a sub array
	void set_data(T const* const, size_t size,size_t gpu);//if 0 treate the array as a full array(initial size) else can be used as a sub array
	void get_data(T*, size_t size=0);//if 0 treate the array as a full array(initial size) else can be used as a sub array
	void get_data(T*, size_t size, size_t gpu);//if 0 treate the array as a full array(initial size) else can be used as a sub array
	void get_new_data(T *&, size_t size=0);//if 0 treate the array as a full array(initial size) else can be used as a sub array
	void get_new_data(T *&, size_t size, size_t gpu);//if 0 treate the array as a full array(initial size) else can be used as a sub array

	void set_data(T const* const, std::vector<size_t> size);
	void get_data(T*, std::vector<size_t> size);
	void get_new_data(T *&, std::vector<size_t> size);
	void set_data(T const* const, std::vector<size_t> size,size_t gpu);
	void get_data(T*, std::vector<size_t> size,size_t gpu);
	void get_new_data(T *&, std::vector<size_t> size,size_t gpu);

	void set_data(mxArray const* const);
	void get_data(mxArray*);
	void get_new_data(mxArray *&, size_t size = 0);
	void get_new_data(mxArray*&, std::vector<size_t> size);
	void set_data(mxArray const* const, size_t gpu);
	void get_data(mxArray*, size_t gpu);
	void get_new_data(mxArray *&, size_t size, size_t gpu);
	void get_new_data(mxArray*&, std::vector<size_t> size, size_t gpu);
	void set_data_multi(mxArray const* const,std::vector<size_t> loaded_gpu);
	void set_data_multi_duplicate(mxArray const* const,std::vector<size_t> loaded_gpu);
	void set_data_multi_duplicate_all(mxArray const* const);
	void get_data_multi(mxArray*,std::vector<size_t> loaded_gpu);
	void get_new_data_multi(mxArray *&,std::vector<size_t> loaded_gpu);
	void get_new_data_multi(mxArray *&,size_t size,std::vector<size_t> loaded_gpu);
	void get_new_data_multi(mxArray*&, std::vector<size_t> size,std::vector<size_t> loaded_gpu);


	void zero();
	void zero_all();
	void zero(size_t gpu);

	T* cuda_ptr();
	T* cuda_ptr(size_t gpu);

	size_t get_size_1D();
	std::vector<size_t> get_size();
private:
	void allocate_buffer(size_t size);
	void de_allocate_buffer();
	mxClassID mex_cuda_class();
	mxComplexity mex_cuda_complexity();
	bool mex_check_type(mxArray const* const matlab_data);
	std::vector<size_t> mex_get_size(mxArray const* const matlab_data);
};

template <class T> CUDA_ARRAY<T>::CUDA_ARRAY(size_t size,std::vector<uint32_t> used_gpu,uint32_t* curr_gpu) {
	gpus=used_gpu;
	current_gpu=curr_gpu;
	cuda_data=std::vector<T*>(gpus.size(),nullptr);
	data_size = {size};
	allocate_buffer(size);
}

template <class T> CUDA_ARRAY<T>::CUDA_ARRAY(std::vector<size_t> size,std::vector<uint32_t> used_gpu,uint32_t* curr_gpu) {
	gpus=used_gpu;
	current_gpu=curr_gpu;
	cuda_data=std::vector<T*>(gpus.size(),nullptr);
	data_size = size;

	size_t size_1D=std::accumulate(begin(size), end(size), 1, std::multiplies<size_t>());
	allocate_buffer(size_1D);
}

template <typename T> T* CUDA_ARRAY<T>::cuda_ptr() {
	return cuda_ptr(*current_gpu);
}
template <typename T> T* CUDA_ARRAY<T>::cuda_ptr(size_t gpu) {
	if (display_debug)mexPrintf("Give pointer %d - gpu %d\n",gpu,gpus[gpu]);
	cudaSetDevice (gpus[gpu]);
	return cuda_data[gpu];
}

template <typename T> mxClassID CUDA_ARRAY<T>::mex_cuda_class() {
	mxClassID classid = mxUNKNOWN_CLASS;
	if (std::is_same<T, unsigned char>::value) classid = mxUINT8_CLASS;
	else if (std::is_same<T, double>::value) classid = mxDOUBLE_CLASS;
	else if (std::is_same<T, float>::value) classid = mxSINGLE_CLASS;
	else if (std::is_same<T, double2>::value) classid = mxDOUBLE_CLASS;
	else if (std::is_same<T, float2>::value) classid = mxSINGLE_CLASS;
	else if (std::is_same<T, int32_t>::value) classid = mxINT32_CLASS;
	else if (std::is_same<T, int16_t>::value) classid = mxINT16_CLASS;
	else if (std::is_same<T, uint32_t>::value) classid = mxUINT32_CLASS;
	else if (std::is_same<T, uint16_t>::value) classid = mxUINT16_CLASS;
	else if (std::is_same<T, bool>::value) classid = mxLOGICAL_CLASS;
	return classid;
}

template <typename T> mxComplexity CUDA_ARRAY<T>::mex_cuda_complexity() {
	if ((std::is_same<T, float2>::value)|| (std::is_same<T, double2>::value))	return mxCOMPLEX;
	else return mxREAL;
}

template <typename T> bool CUDA_ARRAY<T>::mex_check_type(mxArray const* const matlab_data) {
	auto curr_class=mxGetClassID(matlab_data);
	mxComplexity curr_complex = mxREAL;
	if(mxIsComplex(matlab_data)){
		curr_complex = mxCOMPLEX;
	}
	return (curr_class == mex_cuda_class()) && (curr_complex == mex_cuda_complexity());
}
template <typename T> std::vector<size_t> CUDA_ARRAY<T>::mex_get_size(mxArray const* const matlab_data) {
	auto dim_num = mxGetNumberOfDimensions(matlab_data);
	auto dims = mxGetDimensions(matlab_data);
	std::vector<size_t> res(dim_num);
	for (int i = 0; i < dim_num; i++) {
		res[i] = dims[i];
		//mexPrintf("    d: %d\n", res[i]);
	}
	return res;
}

template <class T> CUDA_ARRAY<T>::~CUDA_ARRAY() {
	de_allocate_buffer();
}

template <class T> void CUDA_ARRAY<T>::allocate_buffer(size_t size) {
if (display_debug)mexPrintf("Start buffer allocation\n");
	for (int gpu=0; gpu<gpus.size(); ++gpu){
		if (display_debug)mexPrintf("   %d of gpu %d\n",gpu,gpus[gpu]);
		cudaSetDevice (gpus[gpu]);
		if (display_debug)mexPrintf("Alocate for GPU : %d\n",gpu);
		if (cuda_data[gpu] != nullptr) {
			throw(std::runtime_error("CUDA_ARRAY :: allocate_buffer :: Memory allready set "));
		}
		cuda_data[gpu] = nullptr;
		cudaMalloc((void**)&(cuda_data[gpu]), size*sizeof(T));
		if (cudaGetLastError() != cudaSuccess) {
			throw(std::runtime_error("CUDA_ARRAY :: allocate_buffer :: Cuda error: Failed to allocate "));
			return;
		}
		if (display_debug){
			mexPrintf("buffer was created\n");
			mexPrintf("pointer : %p\n",cuda_data[gpu]);
		}
		cudaMemset(cuda_data[gpu], 0, size*sizeof(T));
		if (cudaGetLastError() != cudaSuccess) {
			throw(std::runtime_error("CUDA_ARRAY :: allocate_buffer :: Cuda error: Failed to Memset "));
			return;
		}
	}
	cudaSetDevice (gpus[*current_gpu]);
	data_size_1D = size;
	if (display_debug)mexPrintf("End buffer allocation\n");
}
template <class T> void CUDA_ARRAY<T>::zero(size_t gpu) {
	if (display_debug)mexPrintf("Set to zero %d gpu %d\n",gpu,gpus[gpu]);
	cudaSetDevice (gpus[gpu]);
	cudaMemset(cuda_data[gpu], 0, get_size_1D() * sizeof(T));
}
template <class T> void CUDA_ARRAY<T>::zero() {
	zero(*current_gpu);
}
template <class T> void CUDA_ARRAY<T>::zero_all() {
	for (int gpu=0; gpu<gpus.size(); ++gpu){
		zero(gpu);
	}
	cudaSetDevice (gpus[*current_gpu]);
}

template <class T> void CUDA_ARRAY<T>::de_allocate_buffer() {
	if (display_debug)mexPrintf("Deallocate buffer \n");
	for (int gpu=0; gpu<gpus.size(); ++gpu){
		if (display_debug)mexPrintf("   %d of gpu %d\n",gpu,gpus[gpu]);
		cudaSetDevice (gpus[gpu]);
		if (cuda_data[gpu] != nullptr){
			if (display_debug){
				mexPrintf("buffer has data\n");
				mexPrintf("pointer : %p\n",cuda_data[gpu]);
			}
			auto code = cudaFree(cuda_data[gpu]);
			if (code != cudaSuccess)
			{
				mexPrintf("GPU kernel assert: %s \n", cudaGetErrorString(code));
				mexErrMsgTxt("CUDA_ARRAY<T>::de_allocate_buffer::An error occured while freeing resources");
			}
			cuda_data[gpu]=nullptr;
		} else {
			if (display_debug)mexPrintf("buffer has no data\n");
		}
	}
	cudaSetDevice (gpus[*current_gpu]);
}

template <class T> size_t CUDA_ARRAY<T>::get_size_1D() {
	return data_size_1D;
}

template <class T> std::vector<size_t> CUDA_ARRAY<T>::get_size() {
	return data_size;
}

template <class T> void CUDA_ARRAY<T>::set_data(T const* const data_cpu, size_t size,size_t gpu) {
	
	if (display_debug)mexPrintf("Set data %d of gpu %d\n",gpu,gpus[gpu]);
	cudaSetDevice (gpus[gpu]);
	if (cuda_data[gpu] == nullptr) {
		throw(std::runtime_error("CUDA_ARRAY :: set_data :: Memory not initialised "+std::to_string(gpu)));
	}
	if (size == 0) {
		size = get_size_1D();
	}
	if (size > get_size_1D()) {
		throw(std::runtime_error("CUDA_ARRAY :: set_data :: requested memory is bigger than available " + std::to_string(size) + "/" + std::to_string(get_size_1D())));
	}
	cudaMemcpy((void*)cuda_data[gpu], (void*)data_cpu, size * sizeof(T), cudaMemcpyHostToDevice);
}
template <class T> void CUDA_ARRAY<T>::set_data(T const* const data_cpu, size_t size) {
	set_data(data_cpu,size,*current_gpu);
}
template <class T> void CUDA_ARRAY<T>::get_data(T* data_cpu, size_t size,size_t gpu) {
	
	if (display_debug)mexPrintf("Get data %d for gpu %d\n",gpu,gpus[gpu]);
	cudaSetDevice (gpus[gpu]);
	if (cuda_data[gpu] == nullptr) {
		throw(std::runtime_error("CUDA_ARRAY :: set_data :: Memory not initialised "+std::to_string(gpu)));
	}
	if (size == 0) {
		size = get_size_1D();
	}
	if (size > get_size_1D()) {
		throw(std::runtime_error("CUDA_ARRAY :: set_data :: requested memory is bigger than available "));
	}
	cudaMemcpy( (void*)data_cpu , (void*)cuda_data[gpu], size * sizeof(T), cudaMemcpyDeviceToHost);
}
template <class T> void CUDA_ARRAY<T>::get_data(T* data_cpu, size_t size) {
	get_data(data_cpu,size,*current_gpu);
}
template <class T> void CUDA_ARRAY<T>::get_new_data(T*& new_array, size_t size) {
	if (size == 0) {
		size = get_size_1D();
	}
	if (size > get_size_1D()) {
		throw(std::runtime_error("CUDA_ARRAY :: set_data :: requested memory is bigger than available "));
	}
	new_array = new T[size];
	get_data(new_array, size);
}
template <class T> void CUDA_ARRAY<T>::get_new_data(T*& new_array, size_t size,size_t gpu) {
	if (size == 0) {
		size = get_size_1D();
	}
	if (size > get_size_1D()) {
		throw(std::runtime_error("CUDA_ARRAY :: set_data :: requested memory is bigger than available "));
	}
	new_array = new T[size];
	get_data(new_array, size,gpu);
}

template <class T> void CUDA_ARRAY<T>::set_data(T const* const data_cpu, std::vector<size_t> size) {
	size_t size_1D = std::accumulate(begin(size), end(size), 1, std::multiplies<size_t>());
	set_data(data_cpu, size_1D);
}
template <class T> void CUDA_ARRAY<T>::get_data(T* data_cpu, std::vector<size_t> size) {
	size_t size_1D = std::accumulate(begin(size), end(size), 1, std::multiplies<size_t>());
	get_data(data_cpu, size_1D);
}
template <class T> void CUDA_ARRAY<T>::get_new_data(T*& new_array, std::vector<size_t> size) {
	size_t size_1D = std::accumulate(begin(size), end(size), 1, std::multiplies<size_t>());
	get_new_data(new_array, size_1D);
}


template <class T> void CUDA_ARRAY<T>::set_data(T const* const data_cpu, std::vector<size_t> size,size_t gpu) {
	size_t size_1D = std::accumulate(begin(size), end(size), 1, std::multiplies<size_t>());
	set_data(data_cpu, size_1D,gpu);
}
template <class T> void CUDA_ARRAY<T>::get_data(T* data_cpu, std::vector<size_t> size,size_t gpu) {
	size_t size_1D = std::accumulate(begin(size), end(size), 1, std::multiplies<size_t>());
	get_data(data_cpu, size_1D,gpu);
}
template <class T> void CUDA_ARRAY<T>::get_new_data(T*& new_array, std::vector<size_t> size,size_t gpu) {
	size_t size_1D = std::accumulate(begin(size), end(size), 1, std::multiplies<size_t>());
	get_new_data(new_array, size_1D,gpu);
}

template <class T> void CUDA_ARRAY<T>::set_data(mxArray const* const matlab_data) {
	if (!mex_check_type(matlab_data)) {
		throw(std::runtime_error("CUDA_ARRAY :: set_data :: Types are not compatible "));
	}
	T const* const data_cpu = static_cast<T const*>(mxGetData(matlab_data));
	if (display_debug) {
		auto size_data = mex_get_size(matlab_data);
		mexPrintf("Upload data, size (%i): /", size_data.size());
		for(auto sz: size_data) mexPrintf("%d/",(int)sz);
		mexPrintf("\n");
	}
	set_data( data_cpu, mex_get_size(matlab_data));
}
template <class T> void CUDA_ARRAY<T>::get_data(mxArray* matlab_data) {
	if (!mex_check_type(matlab_data)) {
		throw(std::runtime_error("CUDA_ARRAY :: set_data :: Types are not compatible "));
	}
	T* data_cpu = static_cast<T*>(mxGetData(matlab_data));
	get_data(data_cpu, mex_get_size(matlab_data));
}
template <class T> void CUDA_ARRAY<T>::get_new_data(mxArray*& matlab_data, size_t size) {
	if (size == 0) {
		get_new_data(matlab_data, get_size());
	}
	else {
		get_new_data(matlab_data, std::vector<size_t>({ size }));
	}
}
template <class T> void CUDA_ARRAY<T>::get_new_data(mxArray*& matlab_data, std::vector<size_t> size) {
	std::vector<mwSize> dims(size.begin(), size.end());
	matlab_data=mxCreateNumericArray((mwSize)dims.size(), dims.data(),mex_cuda_class(), mex_cuda_complexity());
	get_data(matlab_data);
}

template <class T> void CUDA_ARRAY<T>::set_data(mxArray const* const matlab_data,size_t gpu) {
	if (!mex_check_type(matlab_data)) {
		throw(std::runtime_error("CUDA_ARRAY :: set_data :: Types are not compatible "));
	}
	T const* const data_cpu = static_cast<T const*>(mxGetData(matlab_data));
	if (display_debug) {
		auto size_data = mex_get_size(matlab_data);
		mexPrintf("Upload data, size (%i): /", size_data.size());
		for(auto sz: size_data) mexPrintf("%d/",(int)sz);
		mexPrintf("\n");
	}
	set_data( data_cpu, mex_get_size(matlab_data),gpu);
}
template <class T> void CUDA_ARRAY<T>::get_data(mxArray* matlab_data,size_t gpu) {
	if (!mex_check_type(matlab_data)) {
		throw(std::runtime_error("CUDA_ARRAY :: set_data :: Types are not compatible "));
	}
	T* data_cpu = static_cast<T*>(mxGetData(matlab_data));
	get_data(data_cpu, mex_get_size(matlab_data),gpu);
}
template <class T> void CUDA_ARRAY<T>::get_new_data(mxArray*& matlab_data, size_t size,size_t gpu) {
	if (size == 0) {
		get_new_data(matlab_data, get_size(),gpu);
	}
	else {
		get_new_data(matlab_data, std::vector<size_t>({ size }),gpu);
	}
}
template <class T> void CUDA_ARRAY<T>::get_new_data(mxArray*& matlab_data, std::vector<size_t> size,size_t gpu) {
	std::vector<mwSize> dims(size.begin(), size.end());
	matlab_data=mxCreateNumericArray((mwSize)dims.size(), dims.data(),mex_cuda_class(), mex_cuda_complexity());
	get_data(matlab_data,gpu);
}


template <class T> void CUDA_ARRAY<T>::set_data_multi(mxArray const* const matlab_data,std::vector<size_t> loaded_gpu){
	auto size_data = mex_get_size(matlab_data);
	if(loaded_gpu.size()>1){
		if (size_data.back()!=loaded_gpu.size()){
			throw(std::runtime_error("CUDA_ARRAY :: set_data_multi :: size not corresponding to number of gpu : "+std::to_string(size_data.back())+"/"+std::to_string(loaded_gpu.size())));
		}
		size_data.pop_back();
	}
	T const* const data_cpu = static_cast<T const*>(mxGetData(matlab_data));
	size_t linear_size=std::accumulate(begin(size_data), end(size_data), 1, std::multiplies<size_t>());
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		if (loaded_gpu[gpu]>=cuda_data.size()) {
			throw(std::runtime_error("CUDA_ARRAY :: set_data_multi_duplicate :: Out of bound gpu index" ));
		}
		set_data( &(data_cpu[linear_size*gpu]), size_data,loaded_gpu[gpu]);
	}
}
template <class T> void CUDA_ARRAY<T>::set_data_multi_duplicate(mxArray const* const matlab_data,std::vector<size_t> loaded_gpu){
	
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		if (loaded_gpu[gpu]>=cuda_data.size()) {
			throw(std::runtime_error("CUDA_ARRAY :: set_data_multi_duplicate :: Out of bound gpu index" ));
		}
		set_data(matlab_data,loaded_gpu[gpu]);
	}
}
template <class T> void CUDA_ARRAY<T>::set_data_multi_duplicate_all(mxArray const* const matlab_data){
	for (int gpu=0; gpu<gpus.size(); ++gpu){
		set_data(matlab_data,gpu);
	}
}
template <class T> void CUDA_ARRAY<T>::get_data_multi(mxArray* matlab_data,std::vector<size_t> loaded_gpu){
auto size_data = mex_get_size(matlab_data);
	if(loaded_gpu.size()>1){
		if (size_data.back()!=loaded_gpu.size()){
			throw(std::runtime_error("CUDA_ARRAY :: set_data_multi :: size not corresponding to number of gpu : "+std::to_string(size_data.back())+"/"+std::to_string(loaded_gpu.size())));
		}
		size_data.pop_back();
	}
	T* data_cpu = static_cast<T*>(mxGetData(matlab_data));
	size_t linear_size=std::accumulate(begin(size_data), end(size_data), 1, std::multiplies<size_t>());
	for (int gpu=0; gpu<loaded_gpu.size(); ++gpu){
		if (loaded_gpu[gpu]>=cuda_data.size()) {
			throw(std::runtime_error("CUDA_ARRAY :: set_data_multi_duplicate :: Out of bound gpu index" ));
		}
		get_data( &(data_cpu[linear_size*gpu]), size_data,loaded_gpu[gpu]);
	}
}
template <class T> void CUDA_ARRAY<T>::get_new_data_multi(mxArray *& matlab_data,std::vector<size_t> loaded_gpu){
	get_new_data_multi(matlab_data,0,loaded_gpu);
}
template <class T> void CUDA_ARRAY<T>::get_new_data_multi(mxArray *& matlab_data,size_t size,std::vector<size_t> loaded_gpu){
	if (size == 0) {
		get_new_data_multi(matlab_data, get_size(),loaded_gpu);
	}
	else {
		get_new_data_multi(matlab_data, std::vector<size_t>({ size }),loaded_gpu);
	}
}
template <class T> void CUDA_ARRAY<T>::get_new_data_multi(mxArray*& matlab_data, std::vector<size_t> size,std::vector<size_t> loaded_gpu){

	std::vector<mwSize> dims(size.begin(), size.end());
	dims.push_back(loaded_gpu.size());
	matlab_data=mxCreateNumericArray((mwSize)dims.size(), dims.data(),mex_cuda_class(), mex_cuda_complexity());
	get_data_multi(matlab_data,loaded_gpu);

}

#endif