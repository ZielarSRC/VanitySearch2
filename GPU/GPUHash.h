/*
 * Modernized GPUHash.h
 * CUDA 12+ Hash Functions
 */

#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void InitGPUHash();
__device__ void _SHA256_Transform(uint32_t* state, const uint32_t* data);
__device__ void _RIPEMD160_Transform(uint32_t* state, const uint32_t* data);
__device__ void _GetHash160Comp(const uint64_t* px, uint8_t isOdd, uint8_t* hash);
__device__ void _GetHash160(const uint64_t* px, const uint64_t* py, uint8_t* hash);
__device__ void _GetHash160CompSym(const uint64_t* px, uint8_t* hash1, uint8_t* hash2);

#ifdef __cplusplus
}
#endif
