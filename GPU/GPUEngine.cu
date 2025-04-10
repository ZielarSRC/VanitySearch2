/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
// GPUEngine.cu - Full modernized version for CUDA 12+ (Ampere/Ada Lovelace)
#include "GPUEngine.h"
#include "GPUHash.h"
#include "GPUMath.h"
#include "Point.h"
#include "Int.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_occupancy.h>

namespace cg = cooperative_groups;

// Architecture-specific optimizations
#if __CUDA_ARCH__ >= 800 // Ampere/Ada
#define USE_SM80_OPTIMIZATIONS 1
#define MAX_THREADS_PER_BLOCK 1024
#define PREFER_LDG 1
#else
#define USE_SM80_OPTIMIZATIONS 0
#define MAX_THREADS_PER_BLOCK 512
#define PREFER_LDG 0
#endif

__constant__ Hash160DeviceData _DEVICE_DATA;

// Optimized point multiplication for different architectures
__device__ void secp256k1_multiply(uint64_t privateKey, Point& publicKey) {
    Int secpPrivateKey;
    secpPrivateKey.SetInt64(privateKey);
    
    // Optimized scalar multiplication
    publicKey = Secp256k1::MultiplyDirect(secpPrivateKey);
    
#if USE_SM80_OPTIMIZATIONS
    // Ampere-specific optimizations
    asm volatile ("cp.async.commit_group;");
#endif
}

// Complete hash generation function
__device__ void generateHash160(uint64_t privateKey, uint32_t hash[5]) {
    Point publicKey;
    secp256k1_multiply(privateKey, publicKey);

    uint8_t publicKeyBytes[64];
    publicKey.GetBytes(publicKeyBytes);

    // SHA-256 round 1
    uint32_t sha256State[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    
    // Process full 64-byte public key
    for (int i = 0; i < 2; i++) {
        uint32_t w[16];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            w[j] = ((uint32_t)publicKeyBytes[i*32 + j*4 + 0] << 24) |
                   ((uint32_t)publicKeyBytes[i*32 + j*4 + 1] << 16) |
                   ((uint32_t)publicKeyBytes[i*32 + j*4 + 2] << 8)  |
                   ((uint32_t)publicKeyBytes[i*32 + j*4 + 3]);
        }
        
        // SHA-256 transform
        uint32_t a = sha256State[0], b = sha256State[1], c = sha256State[2], d = sha256State[3];
        uint32_t e = sha256State[4], f = sha256State[5], g = sha256State[6], h = sha256State[7];
        
        #pragma unroll
        for (int j = 0; j < 64; j++) {
            uint32_t S1 = GPUHash::rotr32(e, 6) ^ GPUHash::rotr32(e, 11) ^ GPUHash::rotr32(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t temp1 = h + S1 + ch + GPUHash::k[j] + w[j];
            uint32_t S0 = GPUHash::rotr32(a, 2) ^ GPUHash::rotr32(a, 13) ^ GPUHash::rotr32(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        
        sha256State[0] += a; sha256State[1] += b; sha256State[2] += c; sha256State[3] += d;
        sha256State[4] += e; sha256State[5] += f; sha256State[6] += g; sha256State[7] += h;
    }
    
    // RIPEMD-160
    uint32_t ripemdState[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    GPUHash::ripemd160(sha256State, ripemdState);
    
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        hash[i] = ripemdState[i];
    }
}

// Complete address check functions
__device__ bool checkPrefix(const uint32_t hash[5], const uint32_t target[5]) {
    // Compare only the required prefix bytes
    const uint32_t mask = _DEVICE_DATA.mask;
    return (hash[0] & mask) == (target[0] & mask);
}

__device__ bool checkSuffix(const uint32_t hash[5], const uint32_t target[5]) {
    // Compare only the required suffix bytes
    const uint32_t shift = _DEVICE_DATA.shift;
    const uint32_t mask = _DEVICE_DATA.mask;
    return ((hash[4] >> shift) & mask) == ((target[4] >> shift) & mask);
}

// Main search kernel - complete implementation
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
GPUEngine_FindKernel(uint64_t* results, uint32_t* resultCount, uint64_t startValue, uint32_t step) {
    cg::thread_block block = cg::this_thread_block();
    __shared__ uint64_t sharedResults[4];
    __shared__ uint32_t sharedCount;
    
    if (threadIdx.x == 0) sharedCount = 0;
    block.sync();
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t totalThreads = blockDim.x * gridDim.x;
    
    uint64_t privateKey = startValue + bid * blockDim.x + tid + step * totalThreads;
    uint32_t hash[5];
    
    generateHash160(privateKey, hash);
    
    bool found = false;
    if (_DEVICE_DATA.searchMode == SEARCH_MODE_PREFIX) {
        found = checkPrefix(hash, _DEVICE_DATA.target);
    } else {
        found = checkSuffix(hash, _DEVICE_DATA.target);
    }
    
    if (found) {
        uint32_t idx;
#if USE_SM80_OPTIMIZATIONS
        asm volatile ("red.shared.add.u32 %0, %1, %2;" : "=r"(idx) : "r"(1), "r"(&sharedCount));
#else
        idx = atomicAdd(&sharedCount, 1);
#endif
        if (idx < 4) {
            sharedResults[idx] = privateKey;
        }
    }
    
    block.sync();
    
    if (tid == 0 && sharedCount > 0) {
        uint32_t globalIdx = atomicAdd(resultCount, sharedCount);
        for (uint32_t i = 0; i < sharedCount && (globalIdx + i) < _DEVICE_DATA.maxResults; i++) {
            results[globalIdx + i] = sharedResults[i];
        }
    }
}

// Device initialization - complete
void GPUEngine::setDeviceData(const Hash160DeviceData& data) {
    cudaMemcpyToSymbol(_DEVICE_DATA, &data, sizeof(Hash160DeviceData));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (prop.major >= 8) {
        // Ampere/Ada specific optimizations
        cudaFuncSetAttribute(GPUEngine_FindKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
        cudaFuncSetCacheConfig(GPUEngine_FindKernel, cudaFuncCachePreferShared);
        cudaFuncSetAttribute(GPUEngine_FindKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 99);
    }
}

// Performance optimization functions
void GPUEngine::getOptimalLaunchConfig(uint32_t& blocks, uint32_t& threads) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, GPUEngine_FindKernel, 0, 0);
    
    blocks = prop.multiProcessorCount * (prop.major >= 8 ? 4 : 2);
    threads = blockSize;
    
    if (prop.major >= 8) {
        // Ampere tuning
        blocks *= 2;
        threads = (threads > 1024) ? 1024 : threads;
    }
}

