/* Copyright 2013 The MathWorks, Inc. */
//#define DEBUG
#include <stdint.h>
#include <array>

#ifndef PCTDEMO_LIFE_SHMEM_HPP
#define PCTDEMO_LIFE_SHMEM_HPP

int raster_KERNEL(int32_t const * const pBranches, int32_t * const pStep1, int32_t * const pStep2, const int size_Branches, const std::array<int, 3> size_Step1);

#endif
