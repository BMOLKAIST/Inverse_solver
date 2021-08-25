clc;
cd('C:\Users\Administrator\Desktop\HERVE\goldstein_GPU');
mexcuda -output get_lookup_gpu get_lookup_cuda.cu get_lookup_mex.cpp