/* Copyright 2013 The MathWorks, Inc. */
//#define DEBUG
#include <stdint.h>

#ifndef PCTDEMO_LIFE_SHMEM_HPP
#define PCTDEMO_LIFE_SHMEM_HPP

int lookup_KERNEL(int32_t * const pLookupTable, int32_t const * const plookup_z_start, int32_t const * const pResidue, int dim_1_residue, int size_lookup_table);

#endif
