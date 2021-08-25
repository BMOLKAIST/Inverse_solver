clc;
cd('D:\Projects_\_BORN_CONVERGENT\_multiple_scattering_mshh_v2\sub_code\goldstein_GPU');
mexcuda -output raster_branches_gpu raster_branches_cuda.cu raster_branches_mex.cpp