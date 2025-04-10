/*
 * Modernized GPUGroup.h
 * Optimized for VanitySearch2 CUDA 12+
 */

#pragma once
#include <cstdint>

constexpr int GRP_SIZE = 128;

#ifdef __cplusplus
extern "C" {
#endif

void InitGPUGroup(const struct Point* points);
__device__ void GenerateGroup(uint64_t* points);
__device__ void GenerateGroupBatch(uint64_t* points, int batchSize);
__device__ void GenerateKeyGroup(uint64_t* x, uint64_t* y);
__device__ void IncrementKeyGroup(uint64_t* x, uint64_t* y, int increment);
__device__ void GenerateSearchGroup(uint64_t* x, uint64_t* y, 
                                  const uint64_t* baseX, const uint64_t* baseY);

#ifdef __cplusplus
}
#endif
