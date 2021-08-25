clc;
cd('C:\Users\Administrator\Desktop\HERVE\goldstein_GPU');
mexcuda -output remove_dipoles_gpu remove_dipoles_cuda.cu remove_dipoles_mex.cpp