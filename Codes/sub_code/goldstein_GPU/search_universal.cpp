
#ifdef GPU_COMPUTE
__device__
int binarySearch_array_2_size(int arr[], int x, int low, int high)
#else
int binarySearch_array_2_size(int arr[], int x, int low, int high)
#endif
{
	int mid;
	while (low < high) {
		mid = (high + low) / 2;
		if (arr[2 * mid + 1] == x) {
			break;
		}
		else if (arr[2 * mid + 1] > x) {
			high = mid - 1;
		}
		else {
			low = mid + 1;
		}
	}
	mid = (high + low) / 2;
	if (x <= arr[2 * mid + 1])
		return mid;
	else
		return mid + 1;
}
#ifdef GPU_COMPUTE
__device__
void search_in_case(int32_t const * const pLookup, int32_t const * const pResidue, int * const current_shortest, const int square_size, const int box_1, const int box_2, const int start_lookup, const int residue_number_1D, const int lookup_size, const int x_current_residue, const int y_current_residue, const int precompute_number, const int last_dist, const int id_1D) 
#else
void search_in_case(std::vector<unsigned char>& states, int32_t const * const pLookup, int32_t const * const pResidue, int * const current_shortest, const int square_size, const int box_1, const int box_2, const int start_lookup, const int residue_number_1D, const int lookup_size, const int x_current_residue, const int y_current_residue, const int precompute_number, const int last_dist, const int id_1D)
#endif
{
	if (box_1 >= square_size || box_2 >= square_size) {
		//out of the region
		return;
	}
	int case_look_id = start_lookup + box_1 + box_2 * square_size;
	int start_look = pLookup[case_look_id];
	if (start_look == -1) {
		// this box is empty
		return;
	}
	int end_look = residue_number_1D;
	int next_full = 1;
	int val_look;
	while (case_look_id + next_full < lookup_size && end_look == residue_number_1D) {
		val_look = pLookup[case_look_id + next_full];
		if (val_look >= 0) {//because if next box is empty reading would be erronous
			end_look = val_look;
			break;
		}
		next_full++;
	}
	//if(case_look_id + 1 < lookup_size)end_look = pLookup[case_look_id + 1];

	const int x_ID_colomn = 3;
	const int y_ID_colomn = 4;
	const int row_num = 6;

	int dim1, dim2, distance, insertion;
	int exec = 0;
	for (int i = start_look; i < end_look; i++) {
#ifdef GPU_COMPUTE

#else
		if (!(states[i] & mask_active)) {
#endif
			dim1 = pResidue[x_ID_colomn + row_num * i];
			dim2 = pResidue[y_ID_colomn + row_num * i];
			exec++;

			int x_coor = abs(x_current_residue - dim1);
			int y_coor = abs(y_current_residue - dim2);

			//distance = max(x_coor,y_coor);//manhattan distance
			distance = sqrtf(x_coor*x_coor + y_coor * y_coor);//euclidian distance

			int current_last_dist = current_shortest[precompute_number * 2 - 1];
			if (distance < current_last_dist && i != id_1D) {//also check that it is not itself
				insertion = binarySearch_array_2_size(current_shortest, distance, 0, precompute_number - 1);
				// if found an insertion point
				if (insertion < precompute_number) {
					for (int k = precompute_number - 1; k > insertion; k--) {
						current_shortest[2 * k] = current_shortest[2 * (k - 1)];
						current_shortest[2 * k + 1] = current_shortest[2 * (k - 1) + 1];
					}
					current_shortest[2 * insertion] = i;
					current_shortest[2 * insertion + 1] = distance;
				}
			}
#ifdef GPU_COMPUTE

#else
		}
#endif
	}

}

#ifdef GPU_COMPUTE
__global__
void shortest_compute_kernel(int32_t const * const pResidue, int32_t const * const pLookup, int32_t const * const pLookup_z, int32_t const * const pSquareSize, int32_t * const pNearest_res, const int residue_number_1D, const int precompute_number, int32_t const size_3D_1, int32_t const size_3D_2, int32_t const size_3D_3, int lookup_size)
#else
const int precompute_number = 1;
std::array<int, 2*precompute_number> shortest_compute_kernel(const int id_1D, std::vector<unsigned char>& states, int32_t const * const pResidue, int32_t const * const pLookup, int32_t const * const pLookup_z, int32_t const * const pSquareSize, const int residue_number_1D, int lookup_size, const std::array<int, 3>& size_residue_3D)
#endif
{
#ifdef GPU_COMPUTE
	int id_1D = threadIdx.x + blockIdx.x * blockDim.x;
#else
	//precompute only one new near residue
	std::array<int, 2*precompute_number> pNearest_res;
#endif

	if (id_1D < residue_number_1D) { // because can be change to negative if all use when removing dipole along another direction

#ifdef GPU_COMPUTE

#else
		const int size_3D_1 = size_residue_3D[0];
		const int size_3D_2 = size_residue_3D[1];
#endif

		const int case_ID_colomn = 2;
		const int x_ID_colomn = 3;
		const int y_ID_colomn = 4;
		const int z_ID_colomn = 5;

		const int row_num = 6;
		//current residue data
		int case_current_residue = pResidue[case_ID_colomn + row_num * id_1D];
		int x_current_residue = pResidue[x_ID_colomn + row_num * id_1D];
		int y_current_residue = pResidue[y_ID_colomn + row_num * id_1D];
		int z_current_residue = pResidue[z_ID_colomn + row_num * id_1D];
		// starting point of the current z section in the lookup
		const int start_lookup = pLookup_z[z_current_residue];
		//square size at the current z depth
		int square_size = pSquareSize[z_current_residue];
		// get the number of case per side at the current depth
		int case_side_1 = (int)ceilf(((float)size_3D_1) / ((float)square_size));
		int case_side_2 = (int)ceilf(((float)size_3D_2) / ((float)square_size));
		//bicoordinate of case in which the residue is
		int case_current_residue_1 = case_current_residue % square_size;
		int case_current_residue_2 = case_current_residue / square_size;
		int max_border_distance = max(max(square_size - case_current_residue_1, square_size - case_current_residue_2), max(case_current_residue_1, case_current_residue_2));
		//search in adjacent cases
		int * current_shortest = new int[2 * precompute_number];

		//finding algorithm
		int box_search_distance = 0;
		int distance_max = 200000;// just a big number 
		// the maximum distance is the distance to border
		distance_max = min(min(
			x_current_residue, y_current_residue
		), min(
			size_3D_1 - x_current_residue, size_3D_2 - y_current_residue
		));

		//distance_max = distance_max + 5;

		for (int ii = 0; ii < precompute_number; ++ii) {
			current_shortest[ii * 2] = -1;
			current_shortest[ii * 2 + 1] = distance_max;// initialise to -1 --> not found
		}


		int last_dist = distance_max;
		int box_1[4] = { -1,-1,-1,-1 };// the coordinate of the search box along dimenssion 1
		int box_2[4] = { -1,-1,-1,-1 };// the coordinate of the search box along dimenssion 1
		int min_dist_in_search_box[4] = { distance_max,distance_max,distance_max,distance_max };//the min distance in the searched box

		int executed_searches = 0;

		while (box_search_distance < max_border_distance && box_search_distance < square_size) {
			//search in the area for shorter matches
			if (box_search_distance == 0) {
				//search the center
				box_1[0] = case_current_residue_1;
				box_2[0] = case_current_residue_2;
				search_in_case(
#ifdef GPU_COMPUTE

#else
					states,
#endif
					pLookup, pResidue, current_shortest, square_size, box_1[0], box_2[0], start_lookup, residue_number_1D, lookup_size, x_current_residue, y_current_residue, precompute_number, last_dist, id_1D);
				executed_searches++;
			}
			else {
				for (int ii = 1 - box_search_distance; ii <= box_search_distance; ++ii) {
					//top part
					box_1[0] = case_current_residue_1 - box_search_distance;
					box_2[0] = case_current_residue_2 - ii;
					min_dist_in_search_box[0] = max(abs(
						(box_1[0] + 1)*case_side_1 - x_current_residue
					), abs(
					((box_2[0] + (ii > 0 ? 1 : 0))*case_side_2 - y_current_residue)*(ii == 0 ? 0 : 1)
					));

					//right part
					box_1[1] = case_current_residue_1 + ii;
					box_2[1] = case_current_residue_2 - box_search_distance;
					min_dist_in_search_box[1] = max(abs(
						((box_1[1] + (ii > 0 ? 0 : 1))*case_side_1 - x_current_residue)*(ii == 0 ? 0 : 1)
					), abs(
					(box_2[1] + 1)*case_side_2 - y_current_residue
					));

					//left part
					box_1[2] = case_current_residue_1 - ii;
					box_2[2] = case_current_residue_2 + box_search_distance;
					min_dist_in_search_box[2] = max(abs(
						((box_1[2] + (ii > 0 ? 1 : 0))*case_side_1 - x_current_residue)*(ii == 0 ? 0 : 1)
					), abs(
					(box_2[2])*case_side_2 - y_current_residue
					));

					//bottom part
					box_1[3] = case_current_residue_1 + box_search_distance;
					box_2[3] = case_current_residue_2 + ii;
					min_dist_in_search_box[3] = max(abs(
						(box_1[3])*case_side_1 - x_current_residue
					), abs(
					((box_2[3] + (ii > 0 ? 0 : 1))*case_side_2 - y_current_residue)*(ii == 0 ? 0 : 1)
					));

					bool to_eval[4] = { min_dist_in_search_box[0] < last_dist ,min_dist_in_search_box[1] < last_dist ,min_dist_in_search_box[2] < last_dist ,min_dist_in_search_box[3] < last_dist };
					int eval_num = 0;
					while (eval_num < 4) {//this strange loop is to avoid branche which can reduce performances in warps
						while (!to_eval[eval_num] && eval_num < 4) { eval_num++; }
						if (eval_num < 4) {

							search_in_case(
#ifdef GPU_COMPUTE

#else
								states,
#endif
								pLookup, pResidue, current_shortest, square_size, box_1[eval_num], box_2[eval_num], start_lookup, residue_number_1D, lookup_size, x_current_residue, y_current_residue, precompute_number, last_dist, id_1D);
							executed_searches++;

						}
						eval_num++;
					}
					last_dist = current_shortest[precompute_number * 2 - 1];//update the farvest element in the list
				}
			}
			//if all the shortest ar found and next iteration wont give shorter then exit
			last_dist = current_shortest[precompute_number * 2 - 1];//update the farvest element in the list
			if (box_search_distance * case_side_1 >= last_dist && box_search_distance * case_side_2 >= last_dist) {
				break;
			}
			// increase the search radius to fing more
			box_search_distance++;
		}
		//udate the result to the mattrix
#ifdef GPU_COMPUTE
		for (int ii = 0; ii < precompute_number * 2; ++ii) pNearest_res[2 * precompute_number*id_1D + ii] = current_shortest[ii];// return result
#else
		for (int ii = 0; ii < precompute_number * 2; ++ii) pNearest_res[ii] = current_shortest[ii];// no need for offset
#endif
		delete[] current_shortest;
	}
#ifdef GPU_COMPUTE

#else
	return pNearest_res;
#endif
}
