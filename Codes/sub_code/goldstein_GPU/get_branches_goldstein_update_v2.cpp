

#include <stdint.h>
#include<vector>
#include "get_branches_goldstein_struct_v2.hpp"
#include "get_branches_goldstein_update_v2.hpp"
#include "mex.h"
#include <array>
#include <algorithm> //for min max function

using namespace std;//for compatibility of the search with gpu

int get_shortest_id(const active_residue & inspected, int32_t const * const pshortest_list, const int precomputed_distance_number, const int shortest_dim1, const int shortest_dim2) {
	if (inspected.min_distance_idx < precomputed_distance_number) {
		return pshortest_list[inspected.residue*shortest_dim1*shortest_dim2 + inspected.min_distance_idx*shortest_dim1];
	}
	return  inspected.overflow_id;
}
//function to find the shortest element

#include "search_universal.cpp"

// function to update the min distance 
void update_min_distance(active_residue& to_update, std::vector<unsigned char>& states, int32_t const * const presidues, int32_t const * const pshortest_list, int32_t const * const psquare_size, int32_t const * const  plookup_table, int32_t const * const plookup_z_start, const int z_stack_number, const int lookup_length, const int precomputed_distance_number, const int shortest_dim1, const int shortest_dim2, const std::array<int, 3>& size_residue_3D) {
	//first look if can be updated from the precomputed list
	int next_position_min = 1 + to_update.min_distance_idx;
	bool found_valid_residue=false;
	while (!found_valid_residue) {
		if (next_position_min < precomputed_distance_number) {
			//mexPrintf("S1 : %d / %d \n", to_update.residue, next_position_min);
			//if (to_update.residue == -1) {
			//	throw("invalid residue\n"); while (true) { mexEvalString("pause(.001);"); }
			//}
			if (pshortest_list[to_update.residue*shortest_dim1*shortest_dim2 + next_position_min *shortest_dim1]==-1 || 
				next_position_min == precomputed_distance_number-1 || // so that we avoid computing expensive new residue if not necessary
				!(states[pshortest_list[to_update.residue*shortest_dim1*shortest_dim2 + next_position_min*shortest_dim1]] & mask_active)) {// is not active
				//mexPrintf("S2\n");
				to_update.min_distance_idx = next_position_min;
				to_update.min_distance = pshortest_list[to_update.residue*shortest_dim1*shortest_dim2 + next_position_min *shortest_dim1+1];
				found_valid_residue = true;
				//mexPrintf("a \n");
			}
		}// if can not be updated from the precomuted list --> recompute the next nearest
		else {
			next_position_min = precomputed_distance_number;
			to_update.min_distance_idx = next_position_min;
			// find a new residue // costy 
			//mexPrintf("Finding a new shortest residue --> computationally costy --> if persisting try increasing the number of precomputed shortest value .\n");
			//mexPrintf("p \n");
			auto result = shortest_compute_kernel(to_update.residue,  states, presidues, plookup_table, plookup_z_start, psquare_size, states.size(), lookup_length, size_residue_3D);
			//mexPrintf("p2 \n");
			//mexPrintf("RESULT OF SEARCH : %d / %d \n", result[0], result[1]);

			to_update.overflow_id = result[0];
			to_update.min_distance = result[1];

			found_valid_residue = true;
		}
		next_position_min++; //so that if not found looks for the next one
	}
}