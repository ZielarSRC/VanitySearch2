// include/AVX256_SHA256.h
#pragma once
#include <immintrin.h>

class AVX256_SHA256 {
public:
    void hash_4_blocks(__m256i* state, const __m256i* data);
};
