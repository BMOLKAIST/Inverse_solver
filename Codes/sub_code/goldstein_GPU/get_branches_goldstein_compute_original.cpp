#include"get_branches_goldstein_compute.hpp"
#include "get_branches_goldstein_update.hpp"
#include"get_branches_goldstein_struct.hpp"
#include <stdint.h>
#include <vector>
#include <array>
#include "mex.h"
#include <chrono>
#include <thread>


std::vector<branche_coordinates>  get_branches_goldstein_compute(int32_t const * const presidues, int32_t const * const pshortest_list, int32_t const * const psquare_size, int32_t const * const  plookup_table, int32_t const * const plookup_z_start,int residue_number, int z_stack_number, int lookup_length,int precomputed_distance_number, const std::array<int, 3>& size_residue_3D) {
	
	std::vector<branche_coordinates> branche_list;//the resulting branch list

	const int distance_max = 200000;// just a big number 

	//the index corresponding value in presidues
	const int residue_column_number = 6;
	const int residue_residue_idx = 0;
	const int residue_residue_value = 1;
	const int residue_residue_case_id = 2;
	const int residue_residue_x = 3;
	const int residue_residue_y = 4;
	const int residue_residue_z = 5;

	const int shortest_dim2 = precomputed_distance_number;
	const int shortest_dim1 = 2;

	std::vector<unsigned char> residue_state(residue_number , 0);//init all the flags to 0
	std::vector<active_residue> curr_active_residue;//list of residue currently active

	
	for (int i = 0; i < residue_number; i++) { // loop over all the residue and equalise them all
		//mexPrintf("residue = %d .\n", i);
		if (! (residue_state[i] & mask_balanced)) { // residue non balanced --> need to handle it
			//add the first element to active list 
			active_residue start_point;
			start_point.residue = i;
			start_point.parent = 0; 
			start_point.min_distance = 0;//even if not true it will be the first to be searched
			start_point.min_distance_idx = 0;

			curr_active_residue.push_back(start_point);//add to the active vector
			residue_state[start_point.residue] |= mask_active;//set active flag

			int total_load = presidues[i*residue_column_number + residue_residue_value]; // the load to balance

			int current_min_distance = 0;//the curent minimal distance
			int next_min_distance = 0;//the curent minimal distance

			while (total_load != 0) {//while the residue are nor balanced
				//mexPrintf("load = %d / curr_min = %d / next_min = %d .\n",total_load, current_min_distance, next_min_distance);
				current_min_distance = current_min_distance++;// next_min_distance;
				//next_min_distance = distance_max;
				size_t a_i = 0;
				while (a_i < curr_active_residue.size()) {//loop over all the currently active residues
					
					//mexPrintf("loop min_dist = %d / dist_idx = %d / curr min = %d / res a_i = %d / residue = %d \n", curr_active_residue[a_i].min_distance, curr_active_residue[a_i].min_distance_idx, current_min_distance, a_i, curr_active_residue[a_i].residue);
					//if (curr_active_residue[a_i].min_distance > distance_max) {
					//	mexPrintf("insane length"); while (true) { mexEvalString("pause(.001);"); }//throw "insane length";
					//}
					//if (curr_active_residue[a_i].min_distance < 0) {
					//	mexPrintf("insane negative length"); while (true) { mexEvalString("pause(.001);"); }//throw "insane negative length";
					//}
					//mexPrintf("yo\n");
					if (curr_active_residue[a_i].min_distance <= current_min_distance) {//the min distance corresponds to the min distance of the current residue
						
						int shortest_id = get_shortest_id(curr_active_residue[a_i], pshortest_list, precomputed_distance_number, shortest_dim1, shortest_dim2);
						//mexPrintf("L1 a_i = %d \n",a_i);
						//mexPrintf("shortest_id = %d \n", shortest_id);
						if (shortest_id==-1 || !(residue_state[shortest_id] & mask_active)) {//check that it is not active
							
							//mexPrintf("Choosen new ID : %d \n", shortest_id);
							int old_load = total_load;

							if (shortest_id != -1 && !(residue_state[shortest_id] & mask_balanced)) {
								//mexPrintf("val = %d \n", presidues[shortest_id*residue_column_number + residue_residue_value]);
								total_load += presidues[shortest_id*residue_column_number + residue_residue_value];//update the total load
							}
							if (shortest_id == -1){
								total_load = 0;
							}

							//mexPrintf("total_load = %d \n", total_load);

							// add to active list
							active_residue to_add_point;
							to_add_point.residue = shortest_id;
							to_add_point.parent = a_i;
							to_add_point.min_distance = 0;
							to_add_point.min_distance_idx = -1;
							to_add_point.parent_branch = branche_list.size();//the position of the branch
							
							curr_active_residue.push_back(to_add_point);
							
							if (shortest_id != -1) {
								residue_state[shortest_id] |= mask_active;//set active flag
							}

							//mexPrintf("min_idx = %d \n", active_res.residue);
							// add the link
							
							branche_coordinates branch;

							branch.jump_value = 0;
							branch.point_z = presidues[curr_active_residue[a_i].residue *residue_column_number + residue_residue_z];

							if (shortest_id != -1) {
								branch.point_1_x = presidues[shortest_id*residue_column_number + residue_residue_x];
								branch.point_1_y = presidues[shortest_id*residue_column_number + residue_residue_y];
							}
							else {
								branch.point_1_x = -1;
								branch.point_1_y = -1;
							}
							branch.point_2_x = presidues[curr_active_residue[a_i].residue*residue_column_number + residue_residue_x];
							branch.point_2_y = presidues[curr_active_residue[a_i].residue*residue_column_number + residue_residue_y];

							//if (shortest_id != -1) { int dist = sqrt((float)(pow(branch.point_1_x - branch.point_2_x, 2) + pow(branch.point_1_y - branch.point_2_y, 2))); if (dist > 10) { mexPrintf("extraordinary %d  with %d but %d (%d,%d;%d,%d) \n", curr_active_residue[a_i].residue,dist, curr_active_residue[a_i].min_distance, branch.point_1_x, branch.point_1_y, branch.point_2_x, branch.point_2_y); } }

							branche_list.push_back(branch);
							
							// propagate the jump value upward (if the residue isn't equalised)
							if (shortest_id ==-1 || !(residue_state[shortest_id] & mask_balanced)) {
								int jump_to_add;
								if (shortest_id != -1) {
									//mexPrintf("NO border\n");
									jump_to_add = presidues[shortest_id*residue_column_number + residue_residue_value];
								}
								else {
									//mexPrintf("border\n");
									jump_to_add = -old_load;
								}
								//mexPrintf("jump of : %d\n", jump_to_add);
								int parent = curr_active_residue.size() - 1;//the lastly added residue
								while (parent != 0) {//0 is the source so need to go up to the source of the current rsidue group
									//mexPrintf("before jump of : %d\n", branche_list[curr_active_residue[parent].parent_branch].jump_value);
									branche_list[curr_active_residue[parent].parent_branch].jump_value += jump_to_add;
									//mexPrintf("after jump of : %d\n", branche_list[curr_active_residue[parent].parent_branch].jump_value);
									parent = curr_active_residue[parent].parent;
								}
							}
							// update the current min of this element and the added one
							
							//mexPrintf("L3\n");
							
							update_min_distance(curr_active_residue[a_i], residue_state, presidues, pshortest_list, psquare_size, plookup_table, plookup_z_start, z_stack_number, lookup_length, precomputed_distance_number, shortest_dim1, shortest_dim2, size_residue_3D);
							
							//mexPrintf("L3-2\n");

							if (shortest_id != -1) {
								update_min_distance(curr_active_residue.back(), residue_state, presidues, pshortest_list, psquare_size, plookup_table, plookup_z_start, z_stack_number, lookup_length, precomputed_distance_number, shortest_dim1, shortest_dim2, size_residue_3D);
							}
							//mexPrintf("L4\n");

							//update the min distances
							//if (to_add_point.min_distance < current_min_distance) {
							//	current_min_distance = to_add_point.min_distance;
							//}
							//if (to_add_point.min_distance < next_min_distance) {
							//	next_min_distance = to_add_point.min_distance;
							//}
							
						}else {// the shortest residu was active so need to update the nearest list
							//mexPrintf("ABNORMAL UPDATER\n");
							update_min_distance(curr_active_residue[a_i], residue_state, presidues, pshortest_list, psquare_size, plookup_table, plookup_z_start, z_stack_number, lookup_length, precomputed_distance_number, shortest_dim1, shortest_dim2,  size_residue_3D);
						}

						
					}
					//mexPrintf("Loop exit\n");
					if (curr_active_residue[a_i].min_distance < next_min_distance) {//maybee it is the min distance at the next step.
						next_min_distance = curr_active_residue[a_i].min_distance;
					}

					if (total_load == 0) {
						break;//equalised so need to stop
					}

					a_i++;//increment the counter
					
					
				}
				
			}
		}
		//remove the active tag
		for (const auto& res : curr_active_residue) {
			if (res.residue != -1) {
				residue_state[res.residue] &= ~mask_active;//set inactive
				residue_state[res.residue] |= mask_balanced;//set balanced
			}
		}
		//clear the active element list
		curr_active_residue.clear();
		
	}
	
	//mexPrintf("Finished branching\n");mexEvalString("pause(0.001);");

	return branche_list;
}
