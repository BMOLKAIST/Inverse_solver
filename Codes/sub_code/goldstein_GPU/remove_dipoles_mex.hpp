/* Copyright 2013 The MathWorks, Inc. */
//#define DEBUG


#ifndef PCTDEMO_LIFE_SHMEM_HPP
#define PCTDEMO_LIFE_SHMEM_HPP

int dipoles_KERNEL(float* const pResidue, int32_t* const pPositive, float * const pOutArray, float * const pOut2Array, int const * const dims_output, int const dims_Positive);

#endif
