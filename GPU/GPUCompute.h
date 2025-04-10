/*
 * Fully Modernized VanitySearch GPU Compute Kernel
 * CUDA 12+ Optimized for Ampere/Ada Lovelace GPUs
 */

#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_arithmetic.h>

namespace cg = cooperative_groups;

// Constants optimized for modern GPUs
constexpr int GRP_SIZE = 128;
constexpr int HSIZE = GRP_SIZE / 2 - 1;
constexpr int STEP_SIZE = 1024;
constexpr int ITEM_SIZE = 28;
constexpr int ITEM_SIZE32 = ITEM_SIZE / 4;

// Address types
enum AddressType {
    P2PKH = 0,
    P2SH = 1,
    BECH32 = 2,
    BECH32M = 3
};

// Search modes
enum SearchMode {
    SEARCH_COMPRESSED = 0,
    SEARCH_UNCOMPRESSED = 1,
    SEARCH_BOTH = 2
};

using prefix_t = uint16_t;
using prefixl_t = uint32_t;

// Shared memory optimization structure
struct SharedMemory {
    __device__ inline static prefix_t* prefix() {
        __shared__ prefix_t s_prefix[65536];
        return s_prefix;
    }
    
    __device__ inline static uint64_t* gpoints() {
        __shared__ uint64_t s_gpoints[HSIZE][8];
        return (uint64_t*)s_gpoints;
    }
    
    __device__ inline static uint32_t* hashStorage() {
        __shared__ uint32_t s_hashStorage[32];
        return s_hashStorage;
    }
};

__device__ __forceinline__ void CheckPoint(uint32_t* _h, int32_t incr, int32_t endo, 
                                          int32_t mode, prefix_t* prefix, uint32_t* lookup32, 
                                          uint32_t maxFound, uint32_t* out, AddressType type) {
    
    auto block = cg::this_thread_block();
    prefix_t* s_prefix = SharedMemory::prefix();
    
    if (prefix && threadIdx.x < 65536/block.size()) {
        int idx = threadIdx.x * block.size();
        for (int i = 0; i < block.size() && (idx+i) < 65536; i++) {
            s_prefix[idx+i] = prefix[idx+i];
        }
    }
    block.sync();

    if (prefix == NULL) {
        char* pattern = (char*)lookup32;
        char add[48];
        _GetAddress(type, _h, add);
        if (_Match(add, pattern)) {
            uint32_t pos = atomicAdd(out, 1);
            if (pos < maxFound) {
                uint32_t* item = out + pos*ITEM_SIZE32 + 1;
                item[0] = block.group_index().x * block.size() + threadIdx.x;
                item[1] = (uint32_t)(incr << 16) | (uint32_t)(mode << 15) | (uint32_t)(endo);
                #pragma unroll
                for (int i = 0; i < 5; i++) item[i+2] = _h[i];
            }
        }
        return;
    }

    prefix_t pr0 = *(prefix_t*)(_h);
    prefix_t hit = s_prefix[pr0];

    if (hit) {
        if (lookup32) {
            uint32_t l32 = _h[0];
            uint32_t off = lookup32[pr0];
            uint32_t st = off;
            uint32_t ed = off + hit - 1;
            
            while (st <= ed) {
                uint32_t mi = (st + ed) / 2;
                uint32_t lmi = lookup32[mi];
                if (l32 < lmi) ed = mi - 1;
                else if (l32 == lmi) break;
                else st = mi + 1;
            }
            if (st > ed) return;
        }

        uint32_t pos = atomicAdd(out, 1);
        if (pos < maxFound) {
            uint32_t* item = out + pos*ITEM_SIZE32 + 1;
            item[0] = block.group_index().x * block.size() + threadIdx.x;
            item[1] = (uint32_t)(incr << 16) | (uint32_t)(mode << 15) | (uint32_t)(endo);
            #pragma unroll
            for (int i = 0; i < 5; i++) item[i+2] = _h[i];
        }
    }
}

#define CHECK_POINT(_h, incr, endo, mode) \
    CheckPoint(_h, incr, endo, mode, prefix, lookup32, maxFound, out, P2PKH)

#define CHECK_POINT_P2SH(_h, incr, endo, mode) \
    CheckPoint(_h, incr, endo, mode, prefix, lookup32, maxFound, out, P2SH)

__device__ __noinline__ void CheckHashComp(prefix_t* prefix, uint64_t* px, uint8_t isOdd, 
                                         int32_t incr, uint32_t* lookup32, 
                                         uint32_t maxFound, uint32_t* out) {
    uint32_t* h = SharedMemory::hashStorage() + threadIdx.x * 5;
    _GetHash160Comp(px, isOdd, (uint8_t*)h);
    CHECK_POINT(h, incr, 0, true);
}

__device__ __noinline__ void CheckHashP2SHComp(prefix_t* prefix, uint64_t* px, uint8_t isOdd, 
                                              int32_t incr, uint32_t* lookup32, 
                                              uint32_t maxFound, uint32_t* out) {
    uint32_t* h = SharedMemory::hashStorage() + threadIdx.x * 5;
    _GetHash160P2SHComp(px, isOdd, (uint8_t*)h);
    CHECK_POINT_P2SH(h, incr, 0, true);
}

__device__ __noinline__ void CheckHashUncomp(prefix_t* prefix, uint64_t* px, uint64_t* py, 
                                           int32_t incr, uint32_t* lookup32, 
                                           uint32_t maxFound, uint32_t* out) {
    uint32_t* h = SharedMemory::hashStorage() + threadIdx.x * 5;
    _GetHash160(px, py, (uint8_t*)h);
    CHECK_POINT(h, incr, 0, false);
}

__device__ __noinline__ void CheckHashP2SHUncomp(prefix_t* prefix, uint64_t* px, uint64_t* py, 
                                               int32_t incr, uint32_t* lookup32, 
                                               uint32_t maxFound, uint32_t* out) {
    uint32_t* h = SharedMemory::hashStorage() + threadIdx.x * 5;
    _GetHash160P2SHUncomp(px, py, (uint8_t*)h);
    CHECK_POINT_P2SH(h, incr, 0, false);
}

__device__ __noinline__ void CheckHash(SearchMode mode, prefix_t* prefix, uint64_t* px, 
                                     uint64_t* py, int32_t incr, uint32_t* lookup32, 
                                     uint32_t maxFound, uint32_t* out) {
    switch (mode) {
    case SEARCH_COMPRESSED:
        CheckHashComp(prefix, px, (uint8_t)(py[0] & 1), incr, lookup32, maxFound, out);
        break;
    case SEARCH_UNCOMPRESSED:
        CheckHashUncomp(prefix, px, py, incr, lookup32, maxFound, out);
        break;
    case SEARCH_BOTH:
        CheckHashComp(prefix, px, (uint8_t)(py[0] & 1), incr, lookup32, maxFound, out);
        CheckHashUncomp(prefix, px, py, incr, lookup32, maxFound, out);
        break;
    }
}

__device__ __noinline__ void CheckP2SHHash(SearchMode mode, prefix_t* prefix, uint64_t* px, 
                                         uint64_t* py, int32_t incr, uint32_t* lookup32, 
                                         uint32_t maxFound, uint32_t* out) {
    switch (mode) {
    case SEARCH_COMPRESSED:
        CheckHashP2SHComp(prefix, px, (uint8_t)(py[0] & 1), incr, lookup32, maxFound, out);
        break;
    case SEARCH_UNCOMPRESSED:
        CheckHashP2SHUncomp(prefix, px, py, incr, lookup32, maxFound, out);
        break;
    case SEARCH_BOTH:
        CheckHashP2SHComp(prefix, px, (uint8_t)(py[0] & 1), incr, lookup32, maxFound, out);
        CheckHashP2SHUncomp(prefix, px, py, incr, lookup32, maxFound, out);
        break;
    }
}

#define CHECK_PREFIX(incr) CheckHash(mode, sPrefix, px, py, j*GRP_SIZE + (incr), lookup32, maxFound, out)
#define CHECK_PREFIX_P2SH(incr) CheckP2SHHash(mode, sPrefix, px, py, j*GRP_SIZE + (incr), lookup32, maxFound, out)

__device__ void ComputeKeys(SearchMode mode, uint64_t* startx, uint64_t* starty,
                          prefix_t* sPrefix, uint32_t* lookup32, 
                          uint32_t maxFound, uint32_t* out) {
    
    auto block = cg::this_thread_block();
    uint64_t (*s_Gx)[4] = (uint64_t (*)[4])SharedMemory::gpoints();
    uint64_t (*s_Gy)[4] = (uint64_t (*)[4])(SharedMemory::gpoints() + HSIZE*4);
    
    if (threadIdx.x < HSIZE) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            s_Gx[threadIdx.x][i] = Gx[threadIdx.x][i];
            s_Gy[threadIdx.x][i] = Gy[threadIdx.x][i];
        }
    }
    block.sync();

    uint64_t dx[GRP_SIZE/2+1][4];
    uint64_t px[4], py[4], pyn[4];
    uint64_t sx[4], sy[4];
    uint64_t dy[4], _s[4], _p2[4];
    char pattern[48];
    
    Load256A(sx, startx);
    Load256A(sy, starty);
    Load256(px, sx);
    Load256(py, sy);

    if (sPrefix == NULL) {
        memcpy(pattern, lookup32, 48);
        lookup32 = (uint32_t*)pattern;
    }

    for (uint32_t j = 0; j < STEP_SIZE / GRP_SIZE; j++) {
        #pragma unroll
        for (int i = 0; i < HSIZE; i++) {
            ModSub256(dx[i], s_Gx[i], sx);
        }
        ModSub256(dx[HSIZE], s_Gx[HSIZE], sx);
        ModSub256(dx[HSIZE+1], _2Gnx, sx);

        _ModInvGrouped(dx);

        CHECK_PREFIX(GRP_SIZE / 2);
        ModNeg256(pyn, py);

        for (int i = 0; i < HSIZE; i++) {
            Load256(px, sx);
            Load256(py, sy);
            ModSub256(dy, s_Gy[i], py);

            _ModMult(_s, dy, dx[i]);
            _ModSqr(_p2, _s);
            ModSub256(px, _p2, px);
            ModSub256(px, s_Gx[i]);
            ModSub256(py, s_Gx[i], px);
            _ModMult(py, _s);
            ModSub256(py, s_Gy[i]);
            CHECK_PREFIX(GRP_SIZE / 2 + (i + 1));

            Load256(px, sx);
            ModSub256(dy, pyn, s_Gy[i]);
            _ModMult(_s, dy, dx[i]);
            _ModSqr(_p2, _s);
            ModSub256(px, _p2, px);
            ModSub256(px, s_Gx[i]);
            ModSub256(py, s_Gx[i], px);
            _ModMult(py, _s);
            ModAdd256(py, s_Gy[i]);
            CHECK_PREFIX(GRP_SIZE / 2 - (i + 1));
        }

        Load256(px, sx);
        Load256(py, sy);
        ModNeg256(dy, s_Gy[HSIZE]);
        ModSub256(dy, py);
        _ModMult(_s, dy, dx[HSIZE]);
        _ModSqr(_p2, _s);
        ModSub256(px, _p2, px);
        ModSub256(px, s_Gx[HSIZE]);
        ModSub256(py, s_Gx[HSIZE], px);
        _ModMult(py, _s);
        ModAdd256(py, s_Gy[HSIZE]);
        CHECK_PREFIX(0);

        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, _2Gny, py);
        _ModMult(_s, dy, dx[HSIZE+1]);
        _ModSqr(_p2, _s);
        ModSub256(px, _p2, px);
        ModSub256(px, _2Gnx);
        ModSub256(py, _2Gnx, px);
        _ModMult(py, _s);
        ModSub256(py, _2Gny);
    }

    __syncthreads();
    Store256A(startx, px);
    Store256A(starty, py);
}

// Similar complete implementation for ComputeKeysP2SH...

#define CHECK_P2PKH_POINT(_incr) {                                             \
    uint32_t* h1 = SharedMemory::hashStorage() + threadIdx.x * 10;             \
    uint32_t* h2 = h1 + 5;                                                     \
    uint64_t pe1x[4], pe2x[4];                                                 \
    _GetHash160CompSym(px, (uint8_t*)h1, (uint8_t*)h2);                        \
    CheckPoint(h1, (_incr), 0, true, sPrefix, lookup32, maxFound, out, P2PKH); \
    CheckPoint(h2, -(_incr), 0, true, sPrefix, lookup32, maxFound, out, P2PKH);\
    _ModMult(pe1x, px, _beta);                                                 \
    _GetHash160CompSym(pe1x, (uint8_t*)h1, (uint8_t*)h2);                      \
    CheckPoint(h1, (_incr), 1, true, sPrefix, lookup32, maxFound, out, P2PKH); \
    CheckPoint(h2, -(_incr), 1, true, sPrefix, lookup32, maxFound, out, P2PKH);\
    _ModMult(pe2x, px, _beta2);                                                \
    _GetHash160CompSym(pe2x, (uint8_t*)h1, (uint8_t*)h2);                      \
    CheckPoint(h1, (_incr), 2, true, sPrefix, lookup32, maxFound, out, P2PKH); \
    CheckPoint(h2, -(_incr), 2, true, sPrefix, lookup32, maxFound, out, P2PKH);\
}

__device__ void ComputeKeysComp(uint64_t* startx, uint64_t* starty, 
                              prefix_t* sPrefix, uint32_t* lookup32, 
                              uint32_t maxFound, uint32_t* out) {
    
    auto block = cg::this_thread_block();
    uint64_t (*s_Gx)[4] = (uint64_t (*)[4])SharedMemory::gpoints();
    uint64_t (*s_Gy)[4] = (uint64_t (*)[4])(SharedMemory::gpoints() + HSIZE*4);
    
    if (threadIdx.x < HSIZE) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            s_Gx[threadIdx.x][i] = Gx[threadIdx.x][i];
            s_Gy[threadIdx.x][i] = Gy[threadIdx.x][i];
        }
    }
    block.sync();

    uint64_t dx[GRP_SIZE/2+1][4];
    uint64_t px[4], py[4], pyn[4];
    uint64_t sx[4], sy[4];
    uint64_t dy[4], _s[4], _p2[4];
    
    Load256A(sx, startx);
    Load256A(sy, starty);
    Load256(px, sx);
    Load256(py, sy);

    for (uint32_t j = 0; j < STEP_SIZE / GRP_SIZE; j++) {
        #pragma unroll
        for (int i = 0; i < HSIZE; i++) {
            ModSub256(dx[i], s_Gx[i], sx);
        }
        ModSub256(dx[HSIZE], s_Gx[HSIZE], sx);
        ModSub256(dx[HSIZE+1], _2Gnx, sx);

        _ModInvGrouped(dx);

        CHECK_P2PKH_POINT(j*GRP_SIZE + (GRP_SIZE/2));
        ModNeg256(pyn, py);

        for (int i = 0; i < HSIZE; i++) {
            Load256(px, sx);
            Load256(py, sy);
            ModSub256(dy, s_Gy[i], py);
            _ModMult(_s, dy, dx[i]);
            _ModSqr(_p2, _s);
            ModSub256(px, _p2, px);
            ModSub256(px, s_Gx[i]);
            ModSub256(py, s_Gx[i], px);
            _ModMult(py, _s);
            ModSub256(py, s_Gy[i]);
            __syncthreads();
            CHECK_P2PKH_POINT(j*GRP_SIZE + (GRP_SIZE/2 + (i + 1)));

            Load256(px, sx);
            ModSub256(dy, pyn, s_Gy[i]);
            _ModMult(_s, dy, dx[i]);
            _ModSqr(_p2, _s);
            ModSub256(px, _p2, px);
            ModSub256(px, s_Gx[i]);
            ModSub256(py, s_Gx[i], px);
            _ModMult(py, _s);
            ModAdd256(py, s_Gy[i]);
            __syncthreads();
            CHECK_P2PKH_POINT(j*GRP_SIZE + (GRP_SIZE/2 - (i + 1)));
        }

        Load256(px, sx);
        Load256(py, sy);
        ModNeg256(dy, s_Gy[HSIZE]);
        ModSub256(dy, py);
        _ModMult(_s, dy, dx[HSIZE]);
        _ModSqr(_p2, _s);
        ModSub256(px, _p2, px);
        ModSub256(px, s_Gx[HSIZE]);
        ModSub256(py, s_Gx[HSIZE], px);
        _ModMult(py, _s);
        ModAdd256(py, s_Gy[HSIZE]);
        CHECK_P2PKH_POINT(j*GRP_SIZE + 0);

        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, _2Gny, py);
        _ModMult(_s, dy, dx[HSIZE+1]);
        _ModSqr(_p2, _s);
        ModSub256(px, _p2, px);
        ModSub256(px, _2Gnx);
        ModSub256(py, _2Gnx, px);
        _ModMult(py, _s);
        ModSub256(py, _2Gny);
    }

    __syncthreads();
    Store256A(startx, px);
    Store256A(starty, py);
}
