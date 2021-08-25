

#include <stdint.h>
#include<vector>
#include "get_branches_goldstein_struct.hpp"
#include "get_branches_goldstein_update.hpp"
#include "mex.h"
#include <array>

int get_shortest_id(const active_residue & inspected, int32_t const * const pshortest_list, const int precomputed_distance_number, const int shortest_dim1, const int shortest_dim2) {
	if (inspected.min_distance_idx < precomputed_distance_number) {
		return pshortest_list[inspected.residue*shortest_dim1*shortest_dim2 + inspected.min_distance_idx*shortest_dim1];
	}
	return  inspected.overflow_id;
}
//function to find the shortest element



// function to update the min distance 
void update_min_distance(active_residue& to_update, std::vector<unsigned char>& states, int32_t const * const presidues, int32_t const * const pshortest_list, int32_t const * const psquare_size, int32_t const * const  plookup_table, int32_t const * const plookup_z_start, const int z_stack_number, const int lookup_length, const int precomputed_distance_number, const int shortest_dim1, const int shortest_dim2, const std::array<int, 3>& size_residue_3D) {
	//first look if can be updated from the precomputed list
	int next_position_min = 1 + to_update.min_distance_idx;
	bool found_valid_residue=false;
	while (!found_valid_residue) {
		if (next_position_min < precomputed_distance_number) {
			if (!(states[pshortest_list[to_update.residue*shortest_dim1*shortest_dim2 + to_update.min_distance_idx*shortest_dim1]] & mask_active)) {// is not active
				to_update.min_distance_idx = next_position_min;
				to_update.min_distance = pshortest_list[to_update.residue*shortest_dim1*shortest_dim2 + to_update.min_distance_idx*shortest_dim1+1];
				found_valid_residue = true;
			}
		}// if can not be updated from the precomuted list --> recompute the next nearest
		else {
			next_position_min = precomputed_distance_number;
			to_update.min_distance_idx = next_position_min;
			// find a new residue // costy 
			mexPrintf("Finding a new shortest residue --> computationally costy --> if persisting try increasing the number of precomputed shortest value .\n");



			found_valid_residue = true;
		}
	}
}