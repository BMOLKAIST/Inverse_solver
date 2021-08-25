clc;
cd('D:\Projects_\_BORN_CONVERGENT\_multiple_scattering_mshh_v2\sub_code\goldstein_GPU');
mex -output get_branches_goldstein_v2 get_branches_goldstein_v2.cpp get_branches_goldstein_compute_v2.cpp get_branches_goldstein_update_v2.cpp;