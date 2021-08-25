/* Copyright 2013 The MathWorks, Inc. */
//#define DEBUG
#include <stdint.h>

#ifndef PCTDEMO_LIFE_SHMEM_HPP
#define PCTDEMO_LIFE_SHMEM_HPP

int shortest_sort_KERNEL(int32_t const * const pResidue, int32_t const * const pLookup, int32_t const * const pLookup_z, int32_t const * const pSquareSize, int32_t * const pNearest_res, int residue_number_1D, int precompute_number, int32_t* const size_residue_3D, int lookup_size);


#endif
