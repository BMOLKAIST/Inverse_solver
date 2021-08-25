clc;
cd('D:\Projects_\_BORN_CONVERGENT\_multiple_scattering_mshh_v2\sub_code\goldstein_GPU');
mex -output get_branches_goldstein get_branches_goldstein.cpp get_branches_goldstein_compute.cpp get_branches_goldstein_update.cpp;