/*
 * Modernized GPUMath.h
 * CUDA 12+ Elliptic Curve Math
 */

#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void InitGPUMath(const uint64_t* P, const uint64_t* R, 
                const uint64_t* R2, const uint64_t* beta, 
                const uint64_t* beta2);

__device__ void _ModReduce(uint64_t* r, const uint64_t* t);
__device__ void _ModMult(uint64_t* r, const uint64_t* a, const uint64_t* b);
__device__ void _ModSqr(uint64_t* r, const uint64_t* a);
__device__ void _ModInv(uint64_t* r, const uint64_t* a);
__device__ void _ModInvGrouped(uint64_t dx[][4]);
__device__ void ModAdd256(uint64_t* r, const uint64_t* a, const uint64_t* b);
__device__ void ModSub256(uint64_t* r, const uint64_t* a, const uint64_t* b);
__device__ void ModNeg256(uint64_t* r, const uint64_t* a);
__device__ void Load256(uint64_t* r, const uint64_t* a);
__device__ void Load256A(uint64_t* r, const uint64_t* a);
__device__ void Store256(uint64_t* r, const uint64_t* a);
__device__ void Store256A(uint64_t* r, const uint64_t* a);

#ifdef __cplusplus
}
#endif
