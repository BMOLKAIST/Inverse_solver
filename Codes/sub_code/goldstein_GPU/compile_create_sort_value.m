clc;
cd('G:\Moosung Lee\_multiple_scattering_mshh_manuscript_v2\sub_code\goldstein_GPU');
mexcuda -output create_sort_value_gpu create_sort_value_cuda.cu create_sort_value_mex.cpp