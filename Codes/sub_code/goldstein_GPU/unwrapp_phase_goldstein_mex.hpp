/* Copyright 2013 The MathWorks, Inc. */
//#define DEBUG
#include <stdint.h>
#include <array>

#ifndef PCTDEMO_LIFE_SHMEM_HPP
#define PCTDEMO_LIFE_SHMEM_HPP

int unwrapp_KERNEL(float * const pPhase, int32_t const * const pStep1, int32_t const * const pStep2, std::array<int, 3> size_Phase);

#endif
