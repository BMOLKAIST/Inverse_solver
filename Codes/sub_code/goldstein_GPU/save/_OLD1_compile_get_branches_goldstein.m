clc;
cd('C:\Users\Administrator\Desktop\HERVE\goldstein_GPU');
mex -output get_branches_goldstein get_branches_goldstein.cpp get_branches_goldstein_compute.cpp get_branches_goldstein_update.cpp get_branches_goldstein_search.cpp;