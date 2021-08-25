
#include <stdint.h>
#include <vector>
#include <array>
#include"get_branches_goldstein_struct.hpp"

#ifndef BRANCHES_COMPUTE_HPP
#define BRANCHES_COMPUTE_HPP

std::vector<branche_coordinates> get_branches_goldstein_compute(int32_t const * const presidues, int32_t const * const pshortest_list, int32_t const * const psquare_size, int32_t const * const  plookup_table, int32_t const * const plookup_z_start, int residue_number, int z_stack_number, int lookup_length, int precomputed_distance_number, const std::array<int, 3>& size_residue_3D);
#endif