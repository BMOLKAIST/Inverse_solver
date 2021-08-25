clc;
cd('D:\Projects_\_BORN_CONVERGENT\_multiple_scattering_mshh_v2\sub_code\goldstein_GPU');
mexcuda -output remove_dipoles_gpu remove_dipoles_cuda.cu remove_dipoles_mex.cpp