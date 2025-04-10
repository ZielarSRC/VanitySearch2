/*
 * Modernized GPUGroup for VanitySearch2
 * CUDA 12+ Optimized Thread Group Management
 */

#include "GPUGroup.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_arithmetic.h>
#include <vector>
#include <array>

namespace cg = cooperative_groups;

__constant__ uint64_t _Gx[GRP_SIZE/2][4];
__constant__ uint64_t _Gy[GRP_SIZE/2][4];
__constant__ uint64_t _2Gnx[4];
__constant__ uint64_t _2Gny[4];

// Shared memory for group operations
struct GroupSMem {
    __device__ inline static uint64_t* pointBuffer() {
        __shared__ uint64_t buf[GRP_SIZE][4];
        return (uint64_t*)buf;
    }
    
    __device__ inline static uint32_t* indexBuffer() {
        __shared__ uint32_t buf[GRP_SIZE];
        return buf;
    }
};

// Initialize generator points
void InitGPUGroup(const std::vector<Point>& points) {
    std::vector<uint64_t> gx(GRP_SIZE/2 * 4);
    std::vector<uint64_t> gy(GRP_SIZE/2 * 4);
    
    // Convert points to GPU format
    for (int i = 0; i < GRP_SIZE/2; i++) {
        for (int j = 0; j < 4; j++) {
            gx[i*4 + j] = points[i].x.bits64[j];
            gy[i*4 + j] = points[i].y.bits64[j];
        }
    }
    
    // Last point is 2G
    uint64_t _2gnx[4];
    uint64_t _2gny[4];
    for (int j = 0; j < 4; j++) {
        _2gnx[j] = points[GRP_SIZE/2].x.bits64[j];
        _2gny[j] = points[GRP_SIZE/2].y.bits64[j];
    }
    
    cudaMemcpyToSymbol(_Gx, gx.data(), GRP_SIZE/2 * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(_Gy, gy.data(), GRP_SIZE/2 * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(_2Gnx, _2gnx, 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(_2Gny, _2gny, 4 * sizeof(uint64_t));
}

// Optimized point addition for group operations
__device__ void _AddPoints(uint64_t* rx, uint64_t* ry, const uint64_t* px, const uint64_t* py) {
    uint64_t s[4], u[4], t[4];
    
    // Calculate slope s = (py - gy)/(px - gx)
    ModSub256(u, px, _Gx[threadIdx.x]);
    _ModInv(u, u);
    ModSub256(t, py, _Gy[threadIdx.x]);
    _ModMult(s, t, u);
    
    // Calculate rx = s² - px - gx
    _ModSqr(rx, s);
    ModSub256(rx, rx, px);
    ModSub256(rx, rx, _Gx[threadIdx.x]);
    
    // Calculate ry = s*(px - rx) - py
    ModSub256(ry, px, rx);
    _ModMult(ry, ry, s);
    ModSub256(ry, ry, py);
}

// Generate group elements in parallel
__device__ void _GenerateGroup(uint64_t* x, uint64_t* y, int idx) {
    auto block = cg::this_thread_block();
    uint64_t* buf = GroupSMem::pointBuffer();
    
    // Each thread handles one point
    if (threadIdx.x < GRP_SIZE/2) {
        if (idx == 0) {
            // Base point
            Load256(x, _Gx[threadIdx.x]);
            Load256(y, _Gy[threadIdx.x]);
        } else {
            // Add previous point to generator
            _AddPoints(x, y, buf[threadIdx.x*2], buf[threadIdx.x*2 + 1]);
        }
        
        // Store in shared memory
        Store256(buf + threadIdx.x*4, x);
        Store256(buf + threadIdx.x*4 + 2, y);
    }
    block.sync();
    
    // Handle negative elements
    if (threadIdx.x < GRP_SIZE/2) {
        Load256(x, buf + threadIdx.x*4);
        Load256(y, buf + threadIdx.x*4 + 2);
        
        // Negate y coordinate for negative element
        uint64_t ny[4];
        ModNeg256(ny, y);
        
        // Store both positive and negative elements
        uint32_t posIdx = threadIdx.x*2;
        uint32_t negIdx = threadIdx.x*2 + 1;
        
        Store256(buf + posIdx*4, x);
        Store256(buf + posIdx*4 + 2, y);
        Store256(buf + negIdx*4, x);
        Store256(buf + negIdx*4 + 2, ny);
    }
    block.sync();
    
    // Copy to output
    if (threadIdx.x < GRP_SIZE) {
        Load256(x, buf + threadIdx.x*4);
        Load256(y, buf + threadIdx.x*4 + 2);
    }
}

// Optimized group operation with precomputation
__device__ void _GroupOp(uint64_t* x, uint64_t* y, int op) {
    auto block = cg::this_thread_block();
    uint64_t* buf = GroupSMem::pointBuffer();
    uint32_t* indices = GroupSMem::indexBuffer();
    
    // Precompute indices
    if (threadIdx.x == 0) {
        for (int i = 0; i < GRP_SIZE; i++) {
            indices[i] = (i + op) % GRP_SIZE;
        }
    }
    block.sync();
    
    // Reorder points
    if (threadIdx.x < GRP_SIZE) {
        uint64_t tx[4], ty[4];
        Load256(tx, buf + indices[threadIdx.x]*4);
        Load256(ty, buf + indices[threadIdx.x]*4 + 2);
        Store256(x, tx);
        Store256(y, ty);
    }
}

// Generate the complete group
__device__ void GenerateGroup(uint64_t* points) {
    uint64_t x[4], y[4];
    
    // Generate base group
    for (int i = 0; i < GRP_SIZE; i++) {
        _GenerateGroup(x, y, i);
        
        // Store results
        if (threadIdx.x < GRP_SIZE) {
            Store256(points + threadIdx.x*8, x);
            Store256(points + threadIdx.x*8 + 4, y);
        }
        __syncthreads();
    }
    
    // Final group operation
    _GroupOp(x, y, threadIdx.x);
    
    // Store final results
    if (threadIdx.x < GRP_SIZE) {
        Store256(points + threadIdx.x*8, x);
        Store256(points + threadIdx.x*8 + 4, y);
    }
}

// Optimized batch group generation
__device__ void GenerateGroupBatch(uint64_t* points, int batchSize) {
    auto block = cg::this_thread_block();
    uint64_t* buf = GroupSMem::pointBuffer();
    
    for (int b = 0; b < batchSize; b++) {
        // Generate group for this batch item
        uint64_t* out = points + b*GRP_SIZE*8;
        
        for (int i = 0; i < GRP_SIZE; i++) {
            _GenerateGroup(buf + threadIdx.x*4, buf + threadIdx.x*4 + 2, i);
            __syncthreads();
            
            if (threadIdx.x < GRP_SIZE) {
                Store256(out + threadIdx.x*8, buf + threadIdx.x*4);
                Store256(out + threadIdx.x*8 + 4, buf + threadIdx.x*4 + 2);
            }
            __syncthreads();
        }
        
        // Final group operation
        _GroupOp(buf + threadIdx.x*4, buf + threadIdx.x*4 + 2, threadIdx.x);
        
        if (threadIdx.x < GRP_SIZE) {
            Store256(out + threadIdx.x*8, buf + threadIdx.x*4);
            Store256(out + threadIdx.x*8 + 4, buf + threadIdx.x*4 + 2);
        }
        __syncthreads();
    }
}

// Specialized group operations for key generation
__device__ void GenerateKeyGroup(uint64_t* x, uint64_t* y) {
    auto block = cg::this_thread_block();
    uint64_t* buf = GroupSMem::pointBuffer();
    
    // Generate base points
    if (threadIdx.x < GRP_SIZE/2) {
        Load256(x, _Gx[threadIdx.x]);
        Load256(y, _Gy[threadIdx.x]);
        Store256(buf + threadIdx.x*4, x);
        Store256(buf + threadIdx.x*4 + 2, y);
    }
    block.sync();
    
    // Generate all group elements
    for (int i = 1; i < GRP_SIZE/2; i++) {
        if (threadIdx.x < GRP_SIZE/2) {
            _AddPoints(x, y, buf + threadIdx.x*4, buf + threadIdx.x*4 + 2);
            Store256(buf + threadIdx.x*4, x);
            Store256(buf + threadIdx.x*4 + 2, y);
        }
        block.sync();
    }
    
    // Handle negative elements
    if (threadIdx.x < GRP_SIZE/2) {
        Load256(x, buf + threadIdx.x*4);
        Load256(y, buf + threadIdx.x*4 + 2);
        
        uint64_t ny[4];
        ModNeg256(ny, y);
        
        uint32_t posIdx = threadIdx.x*2;
        uint32_t negIdx = threadIdx.x*2 + 1;
        
        Store256(buf + posIdx*4, x);
        Store256(buf + posIdx*4 + 2, y);
        Store256(buf + negIdx*4, x);
        Store256(buf + negIdx*4 + 2, ny);
    }
    block.sync();
    
    // Final 2G point
    if (threadIdx.x == 0) {
        Load256(x, _2Gnx);
        Load256(y, _2Gny);
        Store256(buf + (GRP_SIZE-1)*4, x);
        Store256(buf + (GRP_SIZE-1)*4 + 2, y);
    }
    block.sync();
    
    // Output results
    if (threadIdx.x < GRP_SIZE) {
        Load256(x, buf + threadIdx.x*4);
        Load256(y, buf + threadIdx.x*4 + 2);
    }
}

// Optimized group operation for key increments
__device__ void IncrementKeyGroup(uint64_t* x, uint64_t* y, int increment) {
    auto block = cg::this_thread_block();
    uint64_t* buf = GroupSMem::pointBuffer();
    uint32_t* indices = GroupSMem::indexBuffer();
    
    // Generate base group
    GenerateKeyGroup(x, y);
    
    // Prepare indices for increment
    if (threadIdx.x == 0) {
        for (int i = 0; i < GRP_SIZE; i++) {
            indices[i] = (i + increment) % GRP_SIZE;
        }
    }
    block.sync();
    
    // Apply increment
    if (threadIdx.x < GRP_SIZE) {
        Load256(x, buf + indices[threadIdx.x]*4);
        Load256(y, buf + indices[threadIdx.x]*4 + 2);
    }
}

// Specialized function for key search
__device__ void GenerateSearchGroup(uint64_t* x, uint64_t* y, const uint64_t* baseX, const uint64_t* baseY) {
    auto block = cg::this_thread_block();
    uint64_t* buf = GroupSMem::pointBuffer();
    
    // Add base point to each group element
    if (threadIdx.x < GRP_SIZE/2) {
        uint64_t tx[4], ty[4];
        Load256(tx, _Gx[threadIdx.x]);
        Load256(ty, _Gy[threadIdx.x]);
        
        // Add base point
        uint64_t s[4], u[4], t[4];
        
        ModSub256(u, baseX, tx);
        _ModInv(u, u);
        ModSub256(t, baseY, ty);
        _ModMult(s, t, u);
        
        _ModSqr(x, s);
        ModSub256(x, x, baseX);
        ModSub256(x, x, tx);
        
        ModSub256(y, tx, x);
        _ModMult(y, y, s);
        ModSub256(y, y, ty);
        
        Store256(buf + threadIdx.x*4, x);
        Store256(buf + threadIdx.x*4 + 2, y);
    }
    block.sync();
    
    // Handle negative elements
    if (threadIdx.x < GRP_SIZE/2) {
        Load256(x, buf + threadIdx.x*4);
        Load256(y, buf + threadIdx.x*4 + 2);
        
        uint64_t ny[4];
        ModNeg256(ny, y);
        
        uint32_t posIdx = threadIdx.x*2;
        uint32_t negIdx = threadIdx.x*2 + 1;
        
        Store256(buf + posIdx*4, x);
        Store256(buf + posIdx*4 + 2, y);
        Store256(buf + negIdx*4, x);
        Store256(buf + negIdx*4 + 2, ny);
    }
    block.sync();
    
    // Final 2G point
    if (threadIdx.x == 0) {
        uint64_t s[4], u[4], t[4];
        
        ModSub256(u, baseX, _2Gnx);
        _ModInv(u, u);
        ModSub256(t, baseY, _2Gny);
        _ModMult(s, t, u);
        
        _ModSqr(x, s);
        ModSub256(x, x, baseX);
        ModSub256(x, x, _2Gnx);
        
        ModSub256(y, _2Gnx, x);
        _ModMult(y, y, s);
        ModSub256(y, y, _2Gny);
        
        Store256(buf + (GRP_SIZE-1)*4, x);
        Store256(buf + (GRP_SIZE-1)*4 + 2, y);
    }
    block.sync();
    
    // Output results
    if (threadIdx.x < GRP_SIZE) {
        Load256(x, buf + threadIdx.x*4);
        Load256(y, buf + threadIdx.x*4 + 2);
    }
}