clc;
cd('D:\Projects_\_BORN_CONVERGENT\_multiple_scattering_mshh_v2\sub_code\goldstein_GPU');
mexcuda -output unwrapp_phase_goldstein_gpu unwrapp_phase_goldstein_cuda.cu unwrapp_phase_goldstein_mex.cpp