#include "sha256.h"
#include <immintrin.h>
#include <cstring>

namespace {

alignas(16) const uint32_t INIT_STATE[32] = {
    0x6a09e667, 0x6a09e667, 0x6a09e667, 0x6a09e667,
    0xbb67ae85, 0xbb67ae85, 0xbb67ae85, 0xbb67ae85,
    0x3c6ef372, 0x3c6ef372, 0x3c6ef372, 0x3c6ef372,
    0xa54ff53a, 0xa54ff53a, 0xa54ff53a, 0xa54ff53a,
    0x510e527f, 0x510e527f, 0x510e527f, 0x510e527f,
    0x9b05688c, 0x9b05688c, 0x9b05688c, 0x9b05688c,
    0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab,
    0x5be0cd19, 0x5be0cd19, 0x5be0cd19, 0x5be0cd19
};

inline __m128i Ch(__m128i b, __m128i c, __m128i d) {
    return _mm_xor_si128(_mm_and_si128(b, c), _mm_andnot_si128(b, d));
}

inline __m128i Maj(__m128i b, __m128i c, __m128i d) {
    return _mm_or_si128(_mm_and_si128(b, c), _mm_and_si128(d, _mm_or_si128(b, c)));
}

inline __m128i ROR(__m128i x, int n) {
    return _mm_or_si128(_mm_srli_epi32(x, n), _mm_slli_epi32(x, 32 - n));
}

inline __m128i S0(__m128i x) {
    return _mm_xor_si128(ROR(x, 2), _mm_xor_si128(ROR(x, 13), ROR(x, 22)));
}

inline __m128i S1(__m128i x) {
    return _mm_xor_si128(ROR(x, 6), _mm_xor_si128(ROR(x, 11), ROR(x, 25)));
}

inline __m128i s0(__m128i x) {
    return _mm_xor_si128(ROR(x, 7), _mm_xor_si128(ROR(x, 18), _mm_srli_epi32(x, 3)));
}

inline __m128i s1(__m128i x) {
    return _mm_xor_si128(ROR(x, 17), _mm_xor_si128(ROR(x, 19), _mm_srli_epi32(x, 10)));
}

#define ROUND(a, b, c, d, e, f, g, h, k, w) do { \
    __m128i t1 = _mm_add_epi32(_mm_add_epi32(h, S1(e)), _mm_add_epi32(Ch(e, f, g), _mm_add_epi32(_mm_set1_epi32(k), w)); \
    d = _mm_add_epi32(d, t1); \
    __m128i t2 = _mm_add_epi32(S0(a), Maj(a, b, c)); \
    h = _mm_add_epi32(t1, t2); \
} while (0)

void sha256_sse_transform(__m128i state[8], const uint32_t* data0, const uint32_t* data1, 
                         const uint32_t* data2, const uint32_t* data3) {
    __m128i a = state[0], b = state[1], c = state[2], d = state[3];
    __m128i e = state[4], f = state[5], g = state[6], h = state[7];
    
    __m128i w[16];
    for (int i = 0; i < 16; ++i) {
        w[i] = _mm_set_epi32(data0[i], data1[i], data2[i], data3[i]);
    }
    
    // Compression rounds
    static const uint32_t K[] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    
    for (int i = 0; i < 64; ++i) {
        if (i >= 16) {
            // Message schedule
            w[i & 0xf] = _mm_add_epi32(_mm_add_epi32(s1(w[(i-2) & 0xf]), w[(i-7) & 0xf]), 
                         _mm_add_epi32(s0(w[(i-15) & 0xf]), w[(i-16) & 0xf]));
        }
        
        ROUND(a, b, c, d, e, f, g, h, K[i], w[i & 0xf]);
        
        // Rotate variables
        __m128i temp = a;
        a = h; h = g; g = f; f = e;
        e = d; d = c; c = b; b = temp;
    }
    
    // Update state
    state[0] = _mm_add_epi32(state[0], a);
    state[1] = _mm_add_epi32(state[1], b);
    state[2] = _mm_add_epi32(state[2], c);
    state[3] = _mm_add_epi32(state[3], d);
    state[4] = _mm_add_epi32(state[4], e);
    state[5] = _mm_add_epi32(state[5], f);
    state[6] = _mm_add_epi32(state[6], g);
    state[7] = _mm_add_epi32(state[7], h);
}

} // namespace

void sha256sse_1B(uint32_t *i0, uint32_t *i1, uint32_t *i2, uint32_t *i3,
                 uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3) {
    __m128i state[8];
    memcpy(state, INIT_STATE, sizeof(INIT_STATE));
    
    sha256_sse_transform(state, i0, i1, i2, i3);
    
    // Store results
    const __m128i mask = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    
    for (int i = 0; i < 8; ++i) {
        __m128i s = _mm_shuffle_epi8(state[i], mask);
        if (i < 4) {
            _mm_storeu_si128((__m128i*)(d0 + i*4), s);
            _mm_storeu_si128((__m128i*)(d1 + i*4), s);
            _mm_storeu_si128((__m128i*)(d2 + i*4), s);
            _mm_storeu_si128((__m128i*)(d3 + i*4), s);
        } else {
            _mm_storeu_si128((__m128i*)(d0 + (i-4)*4 + 16), s);
            _mm_storeu_si128((__m128i*)(d1 + (i-4)*4 + 16), s);
            _mm_storeu_si128((__m128i*)(d2 + (i-4)*4 + 16), s);
            _mm_storeu_si128((__m128i*)(d3 + (i-4)*4 + 16), s);
        }
    }
}

void sha256sse_2B(uint32_t *i0, uint32_t *i1, uint32_t *i2, uint32_t *i3,
                 uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3) {
    __m128i state[8];
    memcpy(state, INIT_STATE, sizeof(INIT_STATE));
    
    sha256_sse_transform(state, i0, i1, i2, i3);
    sha256_sse_transform(state, i0 + 16, i1 + 16, i2 + 16, i3 + 16);
    
    // Store results (same as sha256sse_1B)
    const __m128i mask = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    
    for (int i = 0; i < 8; ++i) {
        __m128i s = _mm_shuffle_epi8(state[i], mask);
        if (i < 4) {
            _mm_storeu_si128((__m128i*)(d0 + i*4), s);
            _mm_storeu_si128((__m128i*)(d1 + i*4), s);
            _mm_storeu_si128((__m128i*)(d2 + i*4), s);
            _mm_storeu_si128((__m128i*)(d3 + i*4), s);
        } else {
            _mm_storeu_si128((__m128i*)(d0 + (i-4)*4 + 16), s);
            _mm_storeu_si128((__m128i*)(d1 + (i-4)*4 + 16), s);
            _mm_storeu_si128((__m128i*)(d2 + (i-4)*4 + 16), s);
            _mm_storeu_si128((__m128i*)(d3 + (i-4)*4 + 16), s);
        }
    }
}

void sha256sse_checksum(uint32_t *i0, uint32_t *i1, uint32_t *i2, uint32_t *i3,
                       uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3) {
    __m128i state[8];
    memcpy(state, INIT_STATE, sizeof(INIT_STATE));
    
    // First transform
    sha256_sse_transform(state, i0, i1, i2, i3);
    
    // Prepare second block (padding + length)
    __m128i w0 = _mm_set1_epi32(0x80000000);
    __m128i w1 = _mm_setzero_si128();
    __m128i w2 = _mm_setzero_si128();
    __m128i w3 = _mm_setzero_si128();
    __m128i w4 = _mm_setzero_si128();
    __m128i w5 = _mm_setzero_si128();
    __m128i w6 = _mm_setzero_si128();
    __m128i w7 = _mm_setzero_si128();
    __m128i w8 = _mm_setzero_si128();
    __m128i w9 = _mm_setzero_si128();
    __m128i w10 = _mm_setzero_si128();
    __m128i w11 = _mm_setzero_si128();
    __m128i w12 = _mm_setzero_si128();
    __m128i w13 = _mm_setzero_si128();
    __m128i w14 = _mm_setzero_si128();
    __m128i w15 = _mm_set1_epi32(0x100);
    
    // Second transform
    __m128i a = state[0], b = state[1], c = state[2], d = state[3];
    __m128i e = state[4], f = state[5], g = state[6], h = state[7];
    
    for (int i = 0; i < 64; ++i) {
        if (i >= 16) {
            // Message schedule
            w[i & 0xf] = _mm_add_epi32(_mm_add_epi32(s1(w[(i-2) & 0xf]), w[(i-7) & 0xf]), 
                         _mm_add_epi32(s0(w[(i-15) & 0xf]), w[(i-16) & 0xf]));
        }
        
        static const uint32_t K[] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            // ... (same K array as before)
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };
        
        ROUND(a, b, c, d, e, f, g, h, K[i], w[i & 0xf]);
        
        // Rotate variables
        __m128i temp = a;
        a = h; h = g; g = f; f = e;
        e = d; d = c; c = b; b = temp;
    }
    
    // Final state
    a = _mm_add_epi32(a, state[0]);
    
    // Store checksums
    uint32_t* res = (uint32_t*)&a;
    *((uint32_t*)d0) = bswap_32(res[3]);
    *((uint32_t*)d1) = bswap_32(res[2]);
    *((uint32_t*)d2) = bswap_32(res[1]);
    *((uint32_t*)d3) = bswap_32(res[0]);
}
