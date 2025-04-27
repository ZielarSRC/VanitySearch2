/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 * Copyright (c) 2025 Refactored by Zielar
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

#pragma once

#include <cstdint>
#include <array>

namespace GPUHash {

// Constants
__constant__ constexpr uint32_t SHA256_K[64] = {
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

__constant__ constexpr uint32_t SHA256_IV[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

__constant__ constexpr uint64_t RIPEMD160_SIZE_DESC_32 = 32 << 3;

// SHA-256 Functions
__device__ __forceinline__ uint32_t rotate_right(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t sha256_sigma0(uint32_t x) {
    return rotate_right(x, 2) ^ rotate_right(x, 13) ^ rotate_right(x, 22);
}

__device__ __forceinline__ uint32_t sha256_sigma1(uint32_t x) {
    return rotate_right(x, 6) ^ rotate_right(x, 11) ^ rotate_right(x, 25);
}

__device__ __forceinline__ uint32_t sha256_small_sigma0(uint32_t x) {
    return rotate_right(x, 7) ^ rotate_right(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t sha256_small_sigma1(uint32_t x) {
    return rotate_right(x, 17) ^ rotate_right(x, 19) ^ (x >> 10);
}

__device__ __forceinline__ uint32_t choose(uint32_t x, uint32_t y, uint32_t z) {
    return z ^ (x & (y ^ z));
}

__device__ __forceinline__ uint32_t majority(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) | (z & (x | y));
}

// SHA-256 Round Macro
#define SHA256_ROUND(a, b, c, d, e, f, g, h, k, w) \
    do { \
        uint32_t t1 = h + sha256_sigma1(e) + choose(e, f, g) + k + w; \
        uint32_t t2 = sha256_sigma0(a) + majority(a, b, c); \
        d += t1; \
        h = t1 + t2; \
    } while(0)

// Message Schedule Expansion
__device__ __forceinline__ void expand_message_schedule(uint32_t w[16]) {
    w[0] += sha256_small_sigma1(w[14]) + w[9] + sha256_small_sigma0(w[1]);
    w[1] += sha256_small_sigma1(w[15]) + w[10] + sha256_small_sigma0(w[2]);
    w[2] += sha256_small_sigma1(w[0]) + w[11] + sha256_small_sigma0(w[3]);
    w[3] += sha256_small_sigma1(w[1]) + w[12] + sha256_small_sigma0(w[4]);
    w[4] += sha256_small_sigma1(w[2]) + w[13] + sha256_small_sigma0(w[5]);
    w[5] += sha256_small_sigma1(w[3]) + w[14] + sha256_small_sigma0(w[6]);
    w[6] += sha256_small_sigma1(w[4]) + w[15] + sha256_small_sigma0(w[7]);
    w[7] += sha256_small_sigma1(w[5]) + w[0] + sha256_small_sigma0(w[8]);
    w[8] += sha256_small_sigma1(w[6]) + w[1] + sha256_small_sigma0(w[9]);
    w[9] += sha256_small_sigma1(w[7]) + w[2] + sha256_small_sigma0(w[10]);
    w[10] += sha256_small_sigma1(w[8]) + w[3] + sha256_small_sigma0(w[11]);
    w[11] += sha256_small_sigma1(w[9]) + w[4] + sha256_small_sigma0(w[12]);
    w[12] += sha256_small_sigma1(w[10]) + w[5] + sha256_small_sigma0(w[13]);
    w[13] += sha256_small_sigma1(w[11]) + w[6] + sha256_small_sigma0(w[14]);
    w[14] += sha256_small_sigma1(w[12]) + w[7] + sha256_small_sigma0(w[15]);
    w[15] += sha256_small_sigma1(w[13]) + w[8] + sha256_small_sigma0(w[0]);
}

// SHA-256 Transform
__device__ void sha256_transform(uint32_t state[8], const uint32_t data[16]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    
    uint32_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = data[i];
    }

    // Perform 4 rounds of 16 operations each
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        SHA256_ROUND(a, b, c, d, e, f, g, h, SHA256_K[i*16+0], w[0]);
        SHA256_ROUND(h, a, b, c, d, e, f, g, SHA256_K[i*16+1], w[1]);
        SHA256_ROUND(g, h, a, b, c, d, e, f, SHA256_K[i*16+2], w[2]);
        SHA256_ROUND(f, g, h, a, b, c, d, e, SHA256_K[i*16+3], w[3]);
        SHA256_ROUND(e, f, g, h, a, b, c, d, SHA256_K[i*16+4], w[4]);
        SHA256_ROUND(d, e, f, g, h, a, b, c, SHA256_K[i*16+5], w[5]);
        SHA256_ROUND(c, d, e, f, g, h, a, b, SHA256_K[i*16+6], w[6]);
        SHA256_ROUND(b, c, d, e, f, g, h, a, SHA256_K[i*16+7], w[7]);
        SHA256_ROUND(a, b, c, d, e, f, g, h, SHA256_K[i*16+8], w[8]);
        SHA256_ROUND(h, a, b, c, d, e, f, g, SHA256_K[i*16+9], w[9]);
        SHA256_ROUND(g, h, a, b, c, d, e, f, SHA256_K[i*16+10], w[10]);
        SHA256_ROUND(f, g, h, a, b, c, d, e, SHA256_K[i*16+11], w[11]);
        SHA256_ROUND(e, f, g, h, a, b, c, d, SHA256_K[i*16+12], w[12]);
        SHA256_ROUND(d, e, f, g, h, a, b, c, SHA256_K[i*16+13], w[13]);
        SHA256_ROUND(c, d, e, f, g, h, a, b, SHA256_K[i*16+14], w[14]);
        SHA256_ROUND(b, c, d, e, f, g, h, a, SHA256_K[i*16+15], w[15]);
        
        if (i < 3) {
            expand_message_schedule(w);
        }
    }

    // Update state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// RIPEMD-160 Functions
__device__ __forceinline__ uint32_t rotate_left(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

// RIPEMD-160 Round Functions
__device__ __forceinline__ uint32_t ripemd160_f1(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ y ^ z;
}

__device__ __forceinline__ uint32_t ripemd160_f2(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) | (~x & z);
}

__device__ __forceinline__ uint32_t ripemd160_f3(uint32_t x, uint32_t y, uint32_t z) {
    return (x | ~y) ^ z;
}

__device__ __forceinline__ uint32_t ripemd160_f4(uint32_t x, uint32_t y, uint32_t z) {
    return (x & z) | (~z & y);
}

__device__ __forceinline__ uint32_t ripemd160_f5(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ (y | ~z);
}

// RIPEMD-160 Transform
__device__ void ripemd160_transform(uint32_t state[5], const uint32_t data[16]) {
    uint32_t a1 = state[0], b1 = state[1], c1 = state[2], d1 = state[3], e1 = state[4];
    uint32_t a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;
    uint32_t t;

    // Round 1
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        t = rotate_left(a1 + ripemd160_f1(b1, c1, d1) + data[i], 
                        (uint32_t[]){11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8}[i]) + e1;
        a1 = e1; e1 = d1; d1 = rotate_left(c1, 10); c1 = b1; b1 = t;
    }

    // Round 2
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        t = rotate_left(a1 + ripemd160_f2(b1, c1, d1) + data[(7*i + 0) % 16] + 0x5A827999, 
                        (uint32_t[]){7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12}[i]) + e1;
        a1 = e1; e1 = d1; d1 = rotate_left(c1, 10); c1 = b1; b1 = t;
    }

    // Round 3
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        t = rotate_left(a1 + ripemd160_f3(b1, c1, d1) + data[(3*i + 5) % 16] + 0x6ED9EBA1, 
                        (uint32_t[]){11, 13, 14, 15, 6, 7, 9, 8, 11, 13, 14, 15, 6, 7, 9, 8}[i]) + e1;
        a1 = e1; e1 = d1; d1 = rotate_left(c1, 10); c1 = b1; b1 = t;
    }

    // Round 4
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        t = rotate_left(a1 + ripemd160_f4(b1, c1, d1) + data[(5*i + 1) % 16] + 0x8F1BBCDC, 
                        (uint32_t[]){7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12}[i]) + e1;
        a1 = e1; e1 = d1; d1 = rotate_left(c1, 10); c1 = b1; b1 = t;
    }

    // Round 5
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        t = rotate_left(a1 + ripemd160_f5(b1, c1, d1) + data[i] + 0xA953FD4E, 
                        (uint32_t[]){11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8}[i]) + e1;
        a1 = e1; e1 = d1; d1 = rotate_left(c1, 10); c1 = b1; b1 = t;
    }

    // Update state
    t = state[1] + c1 + d2;
    state[1] = state[2] + d1 + e2;
    state[2] = state[3] + e1 + a2;
    state[3] = state[4] + a1 + b2;
    state[4] = state[0] + b1 + c2;
    state[0] = t;
}

// Hash Computation Functions
__device__ void compute_hash160_compressed(const uint64_t x[4], bool is_odd, uint8_t hash[20]) {
    uint32_t state[8];
    uint32_t data[16] = {0};
    
    // Prepare compressed public key data
    data[0] = __byte_perm(x[3], is_odd ? 0x3 : 0x2, 0x4321);
    data[1] = __byte_perm(x[3], x[2], 0x0765);
    data[2] = __byte_perm(x[2], x[1], 0x0765);
    data[3] = __byte_perm(x[1], x[0], 0x0765);
    data[4] = __byte_perm(x[0], 0x80, 0x0456);
    data[15] = 0x108;

    // SHA-256 first pass
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = SHA256_IV[i];
    }
    sha256_transform(state, data);

    // Prepare for RIPEMD-160
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        data[i] = __byte_perm(state[i], 0, 0x0123); // bswap32
    }
    data[8] = 0x80;
    data[14] = RIPEMD160_SIZE_DESC_32;

    // RIPEMD-160
    uint32_t ripemd_state[5] = {
        0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    };
    ripemd160_transform(ripemd_state, data);

    // Copy result
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        ((uint32_t*)hash)[i] = ripemd_state[i];
    }
}

} // namespace GPUHash
