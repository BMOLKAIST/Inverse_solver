
#include <stdint.h>
#include<vector>

#ifndef BRANCHES_STRUCT_HPP
#define BRANCHES_STRUCT_HPP

struct branche_coordinates {
	int point_1_x;
	int point_1_y;
	int point_2_x;
	int point_2_y;
	int point_z;
	int jump_value;
};
struct branche_id {
	int point_1_id;
	int point_2_id;
	int jump_value;
};
struct active_residue {
	int residue;
	int parent;
	int parent_branch;
	int min_distance;
	int min_distance_idx;//the index of the shortest non used element in the shortest list
	int overflow_id;//if the shortest residue is not in the list it is saved here
};
#endif

