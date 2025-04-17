// AVX256_SHA256.cpp
#include "AVX256_SHA256.h"
#include <immintrin.h>

void AVX256_SHA256::hash_4_keys(const std::array<std::array<uint8_t, 65>, 4>& inputs, 
                               std::array<std::array<uint8_t, 32>, 4>& outputs) {
    __m256i state0, state1, msg0, msg1, msg2, msg3;
    
    // Inicjalizacja stałych SHA-256
    const __m256i init_state = _mm256_setr_epi32(
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    );
    
    // Ładowanie danych wejściowych
    for (int i = 0; i < 4; i++) {
        msg0 = _mm256_loadu_si256((__m256i*)inputs[i].data());
        msg1 = _mm256_loadu_si256((__m256i*)(inputs[i].data() + 32));
        msg2 = _mm256_set1_epi8(inputs[i][64]);
        
        // Procesowanie SHA-256 (pominięte pełne rundy dla czytelności)
        state0 = init_state;
        
        // Pierwsze 16 rund
        for (int round = 0; round < 16; ++round) {
            state0 = _mm256_sha256rnds2_epu32(state0, state1, msg0);
            msg0 = _mm256_sha256msg1_epu32(msg0, msg1);
        }
        
        // Zapis wyników
        _mm256_storeu_si256((__m256i*)outputs[i].data(), state0);
    }
}
