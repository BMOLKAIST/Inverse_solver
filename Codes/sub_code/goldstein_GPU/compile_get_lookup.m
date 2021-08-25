clc;
cd('D:\Projects_\_BORN_CONVERGENT\_multiple_scattering_mshh_v2\sub_code\goldstein_GPU');
mexcuda -output get_lookup_gpu get_lookup_cuda.cu get_lookup_mex.cpp