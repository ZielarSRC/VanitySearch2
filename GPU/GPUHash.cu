/*
 * Modernized GPUHash for VanitySearch2
 * CUDA 12+ Optimized Hashing Functions
 */

#include "GPUHash.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_arithmetic.h>
#include <array>
#include <algorithm>

namespace cg = cooperative_groups;

// SHA-256 and RIPEMD-160 constants in constant memory
__constant__ uint32_t SHA256_K[64];
__constant__ uint32_t RIPEMD160_K[80];
__constant__ uint32_t RIPEMD160_IV[5];

// Shared memory optimization structure
struct HashSMem {
    __device__ inline static uint32_t* sha256WS() {
        __shared__ uint32_t ws[64];
        return ws;
    }
    
    __device__ inline static uint32_t* ripemd160Buffer() {
        __shared__ uint32_t buf[16];
        return buf;
    }
};

// Optimized SHA-256 implementation
__device__ void _SHA256_Transform(uint32_t* state, const uint32_t* data) {
    auto block = cg::this_thread_block();
    uint32_t* W = HashSMem::sha256WS();
    
    // Load data
    if (threadIdx.x < 16) {
        W[threadIdx.x] = data[threadIdx.x];
    }
    block.sync();

    // Message schedule
    #pragma unroll
    for (int t = 16; t < 64; t++) {
        if (threadIdx.x == t % 32) {
            uint32_t s0 = ROTR(W[(t-15)&15], 7) ^ ROTR(W[(t-15)&15], 18) ^ (W[(t-15)&15] >> 3);
            uint32_t s1 = ROTR(W[(t-2)&15], 17) ^ ROTR(W[(t-2)&15], 19) ^ (W[(t-2)&15] >> 10);
            W[t&15] += s0 + W[(t-7)&15] + s1;
        }
        block.sync();
    }

    // Initialize working variables
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    // Compression function
    #pragma unroll
    for (int t = 0; t < 64; t++) {
        uint32_t S1 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + SHA256_K[t] + W[t&15];
        uint32_t S0 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    // Update state
    if (threadIdx.x < 8) {
        state[threadIdx.x] += (threadIdx.x == 0) ? a :
                             (threadIdx.x == 1) ? b :
                             (threadIdx.x == 2) ? c :
                             (threadIdx.x == 3) ? d :
                             (threadIdx.x == 4) ? e :
                             (threadIdx.x == 5) ? f :
                             (threadIdx.x == 6) ? g : h;
    }
}

// Optimized RIPEMD-160 implementation
__device__ void _RIPEMD160_Transform(uint32_t* state, const uint32_t* data) {
    auto block = cg::this_thread_block();
    uint32_t* X = HashSMem::ripemd160Buffer();
    
    // Load data
    if (threadIdx.x < 16) {
        X[threadIdx.x] = data[threadIdx.x];
    }
    block.sync();

    uint32_t al = state[0], ar = state[0];
    uint32_t bl = state[1], br = state[1];
    uint32_t cl = state[2], cr = state[2];
    uint32_t dl = state[3], dr = state[3];
    uint32_t el = state[4], er = state[4];

    // Parallel rounds
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        uint32_t T, rol;
        
        // Left line
        if (i < 16) {
            T = al + (bl ^ cl ^ dl) + X[RIPEMD160_r[i]] + RIPEMD160_K[0];
            rol = RIPEMD160_s[i];
        } else if (i < 32) {
            T = al + (bl & cl | ~bl & dl) + X[RIPEMD160_r[i]] + RIPEMD160_K[1];
            rol = RIPEMD160_s[i];
        } else if (i < 48) {
            T = al + (bl | ~cl ^ dl) + X[RIPEMD160_r[i]] + RIPEMD160_K[2];
            rol = RIPEMD160_s[i];
        } else if (i < 64) {
            T = al + (bl & dl | cl & ~dl) + X[RIPEMD160_r[i]] + RIPEMD160_K[3];
            rol = RIPEMD160_s[i];
        } else {
            T = al + (bl ^ (cl | ~dl)) + X[RIPEMD160_r[i]] + RIPEMD160_K[4];
            rol = RIPEMD160_s[i];
        }
        
        al = el;
        el = dl;
        dl = ROTL(cl, 10);
        cl = bl;
        bl = ROTL(T, rol);

        // Right line
        if (i < 16) {
            T = ar + (br ^ (cr | ~dr)) + X[RIPEMD160_R[i]] + RIPEMD160_K[5];
            rol = RIPEMD160_S[i];
        } else if (i < 32) {
            T = ar + (br & dr | cr & ~dr) + X[RIPEMD160_R[i]] + RIPEMD160_K[6];
            rol = RIPEMD160_S[i];
        } else if (i < 48) {
            T = ar + (br | ~cr ^ dr) + X[RIPEMD160_R[i]] + RIPEMD160_K[7];
            rol = RIPEMD160_S[i];
        } else if (i < 64) {
            T = ar + (br & cr | ~br & dr) + X[RIPEMD160_R[i]] + RIPEMD160_K[8];
            rol = RIPEMD160_S[i];
        } else {
            T = ar + (br ^ cr ^ dr) + X[RIPEMD160_R[i]] + RIPEMD160_K[9];
            rol = RIPEMD160_S[i];
        }
        
        ar = er;
        er = dr;
        dr = ROTL(cr, 10);
        cr = br;
        br = ROTL(T, rol);
    }

    // Final update
    uint32_t t = state[1] + cl + dr;
    state[1] = state[2] + dl + er;
    state[2] = state[3] + el + ar;
    state[3] = state[4] + al + br;
    state[4] = state[0] + bl + cr;
    state[0] = t;
}

// Optimized hash computation for compressed addresses
__device__ void _GetHash160Comp(const uint64_t* px, uint8_t isOdd, uint8_t* hash) {
    uint32_t sha256State[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t ripemd160State[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    
    uint32_t data[16] = {0};
    data[0] = 0x02000000 | (isOdd ? 0x01000000 : 0);
    
    // Convert x coordinate to little-endian
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        data[4 - i] = ((uint32_t*)px)[i];
    }

    // SHA-256 first round
    _SHA256_Transform(sha256State, data);
    
    // Prepare second block (padding)
    data[0] = 0x80000000;
    #pragma unroll
    for (int i = 1; i < 15; i++) {
        data[i] = 0;
    }
    data[15] = 33 * 8; // 33 bytes * 8 bits

    // SHA-256 second round
    _SHA256_Transform(sha256State, data);

    // Prepare for RIPEMD-160
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        data[i] = __byte_perm(sha256State[i], 0, 0x0123); // Endian swap
    }
    data[8] = 0x80000000;
    #pragma unroll
    for (int i = 9; i < 15; i++) {
        data[i] = 0;
    }
    data[15] = 32 * 8; // 32 bytes * 8 bits

    // RIPEMD-160 transform
    _RIPEMD160_Transform(ripemd160State, data);

    // Store result
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        ((uint32_t*)hash)[i] = __byte_perm(ripemd160State[i], 0, 0x0123);
    }
}

// Optimized hash computation for uncompressed addresses
__device__ void _GetHash160(const uint64_t* px, const uint64_t* py, uint8_t* hash) {
    uint32_t sha256State[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t ripemd160State[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    
    uint32_t data[16] = {0};
    data[0] = 0x04000000;
    
    // Convert coordinates to little-endian
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        data[4 - i] = ((uint32_t*)px)[i];
        data[8 - i] = ((uint32_t*)py)[i];
    }

    // SHA-256 first round
    _SHA256_Transform(sha256State, data);
    
    // Prepare second block (padding)
    data[0] = 0x80000000;
    #pragma unroll
    for (int i = 1; i < 15; i++) {
        data[i] = 0;
    }
    data[15] = 65 * 8; // 65 bytes * 8 bits

    // SHA-256 second round
    _SHA256_Transform(sha256State, data);

    // Prepare for RIPEMD-160
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        data[i] = __byte_perm(sha256State[i], 0, 0x0123); // Endian swap
    }
    data[8] = 0x80000000;
    #pragma unroll
    for (int i = 9; i < 15; i++) {
        data[i] = 0;
    }
    data[15] = 32 * 8; // 32 bytes * 8 bits

    // RIPEMD-160 transform
    _RIPEMD160_Transform(ripemd160State, data);

    // Store result
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        ((uint32_t*)hash)[i] = __byte_perm(ripemd160State[i], 0, 0x0123);
    }
}

// Optimized symmetric computation for compressed addresses
__device__ void _GetHash160CompSym(const uint64_t* px, uint8_t* hash1, uint8_t* hash2) {
    uint32_t sha256State[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t ripemd160State[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    
    uint32_t data[16] = {0};
    
    // First hash (odd)
    data[0] = 0x02000000 | 0x01000000;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        data[4 - i] = ((uint32_t*)px)[i];
    }

    _SHA256_Transform(sha256State, data);
    
    data[0] = 0x80000000;
    #pragma unroll
    for (int i = 1; i < 15; i++) data[i] = 0;
    data[15] = 33 * 8;

    _SHA256_Transform(sha256State, data);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        data[i] = __byte_perm(sha256State[i], 0, 0x0123);
    }
    data[8] = 0x80000000;
    #pragma unroll
    for (int i = 9; i < 15; i++) data[i] = 0;
    data[15] = 32 * 8;

    _RIPEMD160_Transform(ripemd160State, data);

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        ((uint32_t*)hash1)[i] = __byte_perm(ripemd160State[i], 0, 0x0123);
    }

    // Second hash (even)
    sha256State[0] = 0x6a09e667; // Reset state
    sha256State[1] = 0xbb67ae85;
    sha256State[2] = 0x3c6ef372;
    sha256State[3] = 0xa54ff53a;
    sha256State[4] = 0x510e527f;
    sha256State[5] = 0x9b05688c;
    sha256State[6] = 0x1f83d9ab;
    sha256State[7] = 0x5be0cd19;

    ripemd160State[0] = 0x67452301;
    ripemd160State[1] = 0xEFCDAB89;
    ripemd160State[2] = 0x98BADCFE;
    ripemd160State[3] = 0x10325476;
    ripemd160State[4] = 0xC3D2E1F0;

    data[0] = 0x02000000;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        data[4 - i] = ((uint32_t*)px)[i];
    }

    _SHA256_Transform(sha256State, data);
    
    data[0] = 0x80000000;
    #pragma unroll
    for (int i = 1; i < 15; i++) data[i] = 0;
    data[15] = 33 * 8;

    _SHA256_Transform(sha256State, data);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        data[i] = __byte_perm(sha256State[i], 0, 0x0123);
    }
    data[8] = 0x80000000;
    #pragma unroll
    for (int i = 9; i < 15; i++) data[i] = 0;
    data[15] = 32 * 8;

    _RIPEMD160_Transform(ripemd160State, data);

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        ((uint32_t*)hash2)[i] = __byte_perm(ripemd160State[i], 0, 0x0123);
    }
}

// Initialize hash constants
void InitGPUHash() {
    // SHA-256 constants
    const uint32_t sha256_k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
        0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
        0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
        0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
        0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
        0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
        0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
        0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
        0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
        0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
        0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
        0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
        0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
        0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
    };
    
    // RIPEMD-160 constants
    const uint32_t ripemd160_k[10] = {
        0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e,
        0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000
    };
    
    const uint32_t ripemd160_iv[5] = {
        0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    };

    cudaMemcpyToSymbol(SHA256_K, sha256_k, sizeof(sha256_k));
    cudaMemcpyToSymbol(RIPEMD160_K, ripemd160_k, sizeof(ripemd160_k));
    cudaMemcpyToSymbol(RIPEMD160_IV, ripemd160_iv, sizeof(ripemd160_iv));
}