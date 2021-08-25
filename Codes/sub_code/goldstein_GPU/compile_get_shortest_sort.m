clc;
cd('D:\Projects_\_BORN_CONVERGENT\_multiple_scattering_mshh_v2\sub_code\goldstein_GPU');
mexcuda -output get_shortest_sort_gpu get_shortest_sort_cuda.cu get_shortest_sort_mex.cpp