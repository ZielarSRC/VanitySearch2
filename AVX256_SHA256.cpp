// src/AVX256_SHA256.cpp
#include "AVX256_SHA256.h"
#include <immintrin.h>

const __m256i SHA256_CONSTANTS[64] = {
    _mm256_set1_epi32(0x428a2f98), _mm256_set1_epi32(0x71374491),
    _mm256_set1_epi32(0xb5c0fbcf), _mm256_set1_epi32(0xe9b5dba5),
    _mm256_set1_epi32(0x3956c25b), _mm256_set1_epi32(0x59f111f1),
    _mm256_set1_epi32(0x923f82a4), _mm256_set1_epi32(0xab1c5ed5),
    _mm256_set1_epi32(0xd807aa98), _mm256_set1_epi32(0x12835b01),
    _mm256_set1_epi32(0x243185be), _mm256_set1_epi32(0x550c7dc3),
    _mm256_set1_epi32(0x72be5d74), _mm256_set1_epi32(0x80deb1fe),
    _mm256_set1_epi32(0x9bdc06a7), _mm256_set1_epi32(0xc19bf174),
    _mm256_set1_epi32(0xe49b69c1), _mm256_set1_epi32(0xefbe4786),
    _mm256_set1_epi32(0x0fc19dc6), _mm256_set1_epi32(0x240ca1cc),
    _mm256_set1_epi32(0x2de92c6f), _mm256_set1_epi32(0x4a7484aa),
    _mm256_set1_epi32(0x5cb0a9dc), _mm256_set1_epi32(0x76f988da),
    _mm256_set1_epi32(0x983e5152), _mm256_set1_epi32(0xa831c66d),
    _mm256_set1_epi32(0xb00327c8), _mm256_set1_epi32(0xbf597fc7),
    _mm256_set1_epi32(0xc6e00bf3), _mm256_set1_epi32(0xd5a79147),
    _mm256_set1_epi32(0x06ca6351), _mm256_set1_epi32(0x14292967),
    _mm256_set1_epi32(0x27b70a85), _mm256_set1_epi32(0x2e1b2138),
    _mm256_set1_epi32(0x4d2c6dfc), _mm256_set1_epi32(0x53380d13),
    _mm256_set1_epi32(0x650a7354), _mm256_set1_epi32(0x766a0abb),
    _mm256_set1_epi32(0x81c2c92e), _mm256_set1_epi32(0x92722c85),
    _mm256_set1_epi32(0xa2bfe8a1), _mm256_set1_epi32(0xa81a664b),
    _mm256_set1_epi32(0xc24b8b70), _mm256_set1_epi32(0xc76c51a3),
    _mm256_set1_epi32(0xd192e819), _mm256_set1_epi32(0xd6990624),
    _mm256_set1_epi32(0xf40e3585), _mm256_set1_epi32(0x106aa070),
    _mm256_set1_epi32(0x19a4c116), _mm256_set1_epi32(0x1e376c08),
    _mm256_set1_epi32(0x2748774c), _mm256_set1_epi32(0x34b0bcb5),
    _mm256_set1_epi32(0x391c0cb3), _mm256_set1_epi32(0x4ed8aa4a),
    _mm256_set1_epi32(0x5b9cca4f), _mm256_set1_epi32(0x682e6ff3),
    _mm256_set1_epi32(0x748f82ee), _mm256_set1_epi32(0x78a5636f),
    _mm256_set1_epi32(0x84c87814), _mm256_set1_epi32(0x8cc70208),
    _mm256_set1_epi32(0x90befffa), _mm256_set1_epi32(0xa4506ceb),
    _mm256_set1_epi32(0xbef9a3f7), _mm256_set1_epi32(0xc67178f2)
};

void AVX256_SHA256::hash_4_blocks(__m256i* state, const __m256i* data) {
    __m256i a, b, c, d, e, f, g, h;
    __m256i w[64];
    
    // Inicjalizacja rejestrów
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    // Rozszerzenie wiadomości
    for (int i = 0; i < 16; ++i) {
        w[i] = _mm256_loadu_si256(data + i);
    }

    for (int i = 16; i < 64; ++i) {
        __m256i s0 = _mm256_sha256msg1_epu32(w[i-15], w[i-14]);
        __m256i s1 = _mm256_sha256msg2_epu32(w[i-2], w[i-7]);
        w[i] = _mm256_add_epi32(w[i-16], _mm256_add_epi32(s0, s1));
    }

    // Główna pętla SHA-256
    for (int i = 0; i < 64; ++i) {
        __m256i t1 = _mm256_add_epi32(
            _mm256_add_epi32(h, _mm256_sha256sig1_epu32(e)),
            _mm256_add_epi32(
                _mm256_sha256rnds2_epu32(e, f, g),
                _mm256_add_epi32(w[i], SHA256_CONSTANTS[i])
            )
        );
        __m256i t2 = _mm256_sha256sig0_epu32(a);
        h = g;
        g = f;
        f = e;
        e = _mm256_add_epi32(d, t1);
        d = c;
        c = b;
        b = a;
        a = _mm256_add_epi32(t1, t2);
    }

    // Aktualizacja stanu
    state[0] = _mm256_add_epi32(state[0], a);
    state[1] = _mm256_add_epi32(state[1], b);
    state[2] = _mm256_add_epi32(state[2], c);
    state[3] = _mm256_add_epi32(state[3], d);
    state[4] = _mm256_add_epi32(state[4], e);
    state[5] = _mm256_add_epi32(state[5], f);
    state[6] = _mm256_add_epi32(state[6], g);
    state[7] = _mm256_add_epi32(state[7], h);
}
