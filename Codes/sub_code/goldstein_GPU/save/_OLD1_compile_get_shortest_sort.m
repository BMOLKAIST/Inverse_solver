clc;
cd('C:\Users\Administrator\Desktop\HERVE\goldstein_GPU');
mexcuda -output get_shortest_sort_gpu get_shortest_sort_cuda.cu get_shortest_sort_mex.cpp