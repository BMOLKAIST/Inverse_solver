
#include <stdint.h>
#include<vector>
#include"get_branches_goldstein_struct_v2.hpp"
#include <array>

#ifndef BRANCHES_UPDATE_HPP
#define BRANCHES_UPDATE_HPP

constexpr unsigned char mask_balanced{ 1 << 0 }; // 0000 0001 // is balanced ?
constexpr unsigned char mask_active{ 1 << 1 }; // 0000 0010 // is active  ?
constexpr unsigned char mask2{ 1 << 2 }; // 0000 0100 
constexpr unsigned char mask3{ 1 << 3 }; // 0000 1000
constexpr unsigned char mask4{ 1 << 4 }; // 0001 0000
constexpr unsigned char mask5{ 1 << 5 }; // 0010 0000
constexpr unsigned char mask6{ 1 << 6 }; // 0100 0000
constexpr unsigned char mask7{ 1 << 7 }; // 1000 0000

//std::cout << "bit 1 is " << ((flags & mask1) ? "on\n" : "off\n");
//flags |= mask1; // turn on bit 1
//flags &= ~mask2; // turn off bit 2

void update_min_distance(active_residue& to_update, std::vector<unsigned char>& states, int32_t const * const presidues, int32_t const * const pshortest_list, int32_t const * const psquare_size, int32_t const * const  plookup_table, int32_t const * const plookup_z_start, const int z_stack_number, const int lookup_length, const int precomputed_distance_number, const int shortest_dim1, const int shortest_dim2, const std::array<int, 3>& size_residue_3D);
int get_shortest_id(const active_residue & inspected, int32_t const * const pshortest_list, const int precomputed_distance_number, const int shortest_dim1, const int shortest_dim2);

#endif

