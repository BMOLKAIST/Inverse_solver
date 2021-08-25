clc;
cd('Z:\MSLee\_multiple_scattering_mshh_manuscript_v2\sub_code\goldstein_GPU');
mexcuda -output create_sort_value_gpu create_sort_value_cuda.cu create_sort_value_mex.cpp;
mex -output get_branches_goldstein get_branches_goldstein.cpp get_branches_goldstein_compute.cpp get_branches_goldstein_update.cpp;
mex -output get_branches_goldstein_original get_branches_goldstein.cpp get_branches_goldstein_compute_original.cpp get_branches_goldstein_update.cpp;
mex -output get_branches_goldstein_v2 get_branches_goldstein_v2.cpp get_branches_goldstein_compute_v2.cpp get_branches_goldstein_update_v2.cpp;
mexcuda -output get_lookup_gpu get_lookup_cuda.cu get_lookup_mex.cpp;
mexcuda -output get_residue_gpu get_residue_cuda.cu get_residue_mex.cpp;
mexcuda -output get_shortest_sort_gpu get_shortest_sort_cuda.cu get_shortest_sort_mex.cpp;
mexcuda -output raster_branches_gpu raster_branches_cuda.cu raster_branches_mex.cpp;
mexcuda -output raster_branches_display_gpu raster_branches_display_cuda.cu raster_branches_mex.cpp;
mexcuda -output remove_dipoles_gpu remove_dipoles_cuda.cu remove_dipoles_mex.cpp;
mexcuda -output unwrapp_phase_goldstein_gpu unwrapp_phase_goldstein_cuda.cu unwrapp_phase_goldstein_mex.cpp;




