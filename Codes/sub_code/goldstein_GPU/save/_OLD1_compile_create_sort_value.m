clc;
cd('C:\Users\Administrator\Desktop\HERVE\goldstein_GPU');
mexcuda -output create_sort_value_gpu create_sort_value_cuda.cu create_sort_value_mex.cpp