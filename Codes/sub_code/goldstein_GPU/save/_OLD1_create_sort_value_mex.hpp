/* Copyright 2013 The MathWorks, Inc. */
//#define DEBUG
#include <stdint.h>

#ifndef PCTDEMO_LIFE_SHMEM_HPP
#define PCTDEMO_LIFE_SHMEM_HPP

int create_sort_KERNEL(int32_t const* const pResidue, int32_t const* const pSquareSize, int32_t* const pCaseID, int32_t* const pDim1ID, int32_t* const pDim2ID, int32_t* const size_residue_3D, int32_t size_residue_1D);

#endif
