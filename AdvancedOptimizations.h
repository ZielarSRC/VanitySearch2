// AdvancedOptimizations.h
#pragma once
#include <immintrin.h>

class MontgomeryLadder {
public:
    void scalar_multiply(Point& result, const Int& scalar, const Point& point) {
        // Bezpieczna implementacja Montgomery Ladder
        Point r0 = Point::Infinity();
        Point r1 = point;
        
        for (int i = 255; i >= 0; i--) {
            if (scalar.bit(i)) {
                add_points(r0, r0, r1);
                double_point(r1, r1);
            } else {
                add_points(r1, r1, r0);
                double_point(r0, r0);
            }
        }
        result = r0;
    }
};

class SIMDKeyGenerator {
public:
    void generate_4_keys(Int* keys) {
        __m256i rand_data = _mm256_loadu_si256(
            (__m256i*)rdrand256());
        
        for (int i = 0; i < 4; i++) {
            _mm256_storeu_si256(
                (__m256i*)keys[i].bits64,
                _mm256_add_epi64(rand_data, _mm256_set1_epi64x(i)));
        }
    }
    
private:
    const uint64_t* rdrand256() {
        alignas(32) static uint64_t buffer[4];
        for (int i = 0; i < 4; i++) {
            _rdrand64_step(&buffer[i]);
        }
        return buffer;
    }
};

class PrecomputedTables {
private:
    std::vector<Point> precomp_g;
    std::vector<Point> precomp_g_128;
    
public:
    PrecomputedTables() {
        // Prekomputacja 256 punktów dla szybkiego mnożenia
        Point current = Secp256k1::G();
        for (int i = 0; i < 256; i++) {
            precomp_g.push_back(current);
            double_point(current, current);
        }
        
        // Prekomputacja dla 128-bitowej optymalizacji
        current = Secp256k1::G();
        for (int i = 0; i < 128; i++) {
            precomp_g_128.push_back(current);
            for (int j = 0; j < 64; j++) {
                double_point(current, current);
            }
        }
    }
    
    void fast_multiply(Point& result, const Int& scalar) {
        result = Point::Infinity();
        for (int i = 0; i < 256; i++) {
            if (scalar.bit(i)) {
                add_points(result, result, precomp_g[i]);
            }
        }
    }
};