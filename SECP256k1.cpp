#include "SECP256k1.h"
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>
#include <limits>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// Stałe krzywej secp256k1
const SECP256k1::uint256_t SECP256k1::P = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

const SECP256k1::uint256_t SECP256k1::Gx = {
    0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07,
    0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798
};

const SECP256k1::uint256_t SECP256k1::Gy = {
    0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8,
    0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8
};

const SECP256k1::uint256_t SECP256k1::Beta = {
    0x7AE96A2B, 0x657C0710, 0x6E64479E, 0xAC3434E9,
    0x9CF04975, 0x12F58995, 0xC1396C28, 0x7AEA2303
};

std::array<SECP256k1::uint256_t, SECP256k1::PRECOMP_TABLE_SIZE> SECP256k1::precompTable;
std::array<SECP256k1::uint256_t, SECP256k1::PRECOMP_TABLE_SIZE> SECP256k1::precompTableY;

//================================================
// Implementacje operacji modularnych
//================================================

void SECP256k1::ModReduce(uint256_t& val) {
    uint256_t p = P;
    while (val >= p) {
        uint32_t borrow = 0;
        uint256_t tmp;
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = static_cast<uint64_t>(val.data[i]) 
                          - static_cast<uint64_t>(p.data[i]) 
                          - borrow;
            borrow = (diff >> 63) & 1;
            tmp.data[i] = static_cast<uint32_t>(diff);
        }
        if (borrow == 0) val = tmp;
    }
}

void SECP256k1::ModAdd(uint256_t& a, const uint256_t& b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t sum = static_cast<uint64_t>(a.data[i]) 
                     + static_cast<uint64_t>(b.data[i]) 
                     + carry;
        a.data[i] = static_cast<uint32_t>(sum);
        carry = sum >> 32;
    }
    ModReduce(a);
}

void SECP256k1::ModSub(uint256_t& a, const uint256_t& b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t diff = static_cast<uint64_t>(a.data[i]) 
                      - static_cast<uint64_t>(b.data[i]) 
                      - borrow;
        borrow = (diff >> 63) & 1;
        a.data[i] = static_cast<uint32_t>(diff);
    }
    if (borrow) ModAdd(a, P);
    ModReduce(a);
}

void SECP256k1::ModMul(const uint256_t& a, const uint256_t& b, uint256_t& result) {
    uint64_t product[16] = {0};
    for (int i = 0; i < 8; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; ++j) {
            product[i + j] += static_cast<uint64_t>(a.data[i]) 
                            * static_cast<uint64_t>(b.data[j]) 
                            + carry;
            carry = product[i + j] >> 32;
            product[i + j] &= 0xFFFFFFFF;
        }
        product[i + 8] = carry;
    }
    
    // Redukcja przy użyciu P
    for (int i = 15; i >= 8; --i) {
        uint64_t val = product[i];
        product[i - 8] += val * 0x1000003D1; // Stała dla secp256k1
        for (int j = i - 7; j <= i; ++j) {
            product[j] += (val << 32);
            val = product[j] >> 32;
            product[j] &= 0xFFFFFFFF;
        }
    }
    
    for (int i = 0; i < 8; ++i) {
        result.data[i] = static_cast<uint32_t>(product[i]);
    }
    ModReduce(result);
}

void SECP256k1::ModInverse(const uint256_t& a, uint256_t& result) {
    uint256_t u = a, v = P;
    uint256_t x1 = {1}, x2 = {0};
    
    while (u != uint256_t{1}) {
        uint256_t q, r, t;
        
        // Dzielenie v przez u
        uint256_t tmp = v;
        uint32_t shift = 0;
        while (tmp >= u && shift < 256) {
            tmp = tmp >> 1;
            shift++;
        }
        q = uint256_t{1} << shift;
        r = v - (u * q);
        
        t = x2 + (q * x1);
        x2 = x1;
        x1 = t;
        v = u;
        u = r;
    }
    result = x1;
    ModReduce(result);
}

//================================================
// Operacje na punktach krzywej
//================================================

void SECP256k1::PointDouble(uint256_t& x, uint256_t& y) {
    uint256_t lambda;
    uint256_t numerator, denominator;
    
    ModMul(x, x, numerator);
    ModAdd(numerator, numerator);
    ModAdd(numerator, numerator); // 3x^2
    
    ModAdd(y, y); // 2y
    ModInverse(y, denominator);
    
    ModMul(numerator, denominator, lambda);
    
    uint256_t new_x, new_y;
    ModMul(lambda, lambda, new_x);
    ModSub(new_x, x);
    ModSub(new_x, x);
    
    ModSub(x, new_x, new_y);
    ModMul(lambda, new_y, new_y);
    ModSub(new_y, y);
    
    x = new_x;
    y = new_y;
}

void SECP256k1::PointAddJacobian(const uint256_t& x1, const uint256_t& y1, const uint256_t& z1,
                                const uint256_t& x2, const uint256_t& y2, const uint256_t& z2,
                                uint256_t& outX, uint256_t& outY, uint256_t& outZ) {
    if (z1 == uint256_t{0}) {
        outX = x2;
        outY = y2;
        outZ = z2;
        return;
    }
    if (z2 == uint256_t{0}) {
        outX = x1;
        outY = y1;
        outZ = z1;
        return;
    }
    
    uint256_t z1_sq, z2_sq, u1, u2, s1, s2;
    ModMul(z1, z1, z1_sq);
    ModMul(z2, z2, z2_sq);
    ModMul(x1, z2_sq, u1);
    ModMul(x2, z1_sq, u2);
    ModMul(y1, z2_sq, s1); ModMul(s1, z2, s1);
    ModMul(y2, z1_sq, s2); ModMul(s2, z1, s2);
    
    if (u1 == u2) {
        if (s1 != s2) {
            outX = outY = outZ = uint256_t{0};
            return;
        }
        PointDoubleJacobian(outX, outY, outZ);
        return;
    }
    
    uint256_t h, r, h_sq, h_cu;
    ModSub(u2, u1, h);
    ModSub(s2, s1, r);
    
    ModMul(h, h, h_sq);
    ModMul(h_sq, h, h_cu);
    
    uint256_t tmp;
    ModMul(u1, h_sq, tmp);
    
    ModMul(r, r, outX);
    ModSub(outX, h_cu);
    ModSub(outX, tmp);
    ModSub(outX, tmp);
    
    ModSub(tmp, outX, outY);
    ModMul(outY, r, outY);
    ModMul(s1, h_cu, tmp);
    ModSub(outY, tmp);
    
    ModMul(z1, z2, outZ);
    ModMul(outZ, h, outZ);
}

//================================================
// Implementacja CUDA
//================================================

#ifdef USE_CUDA

__global__ void CudaMultiplyKernel(const uint32_t* d_gx, const uint32_t* d_gy,
                                  const SECP256k1::uint256_t* d_keys,
                                  uint32_t* d_results, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    SECP256k1::uint256_t key = d_keys[idx];
    SECP256k1::uint256_t x, y;
    
    // Implementacja Montgomery Ladder
    SECP256k1::uint256_t xj = {0}, yj = {0}, zj = {0};
    xj.data[0] = 1; // Punkt neutralny
    
    for (int i = 255; i >= 0; --i) {
        int byte_idx = i / 32;
        int bit_idx = i % 32;
        bool bit = (key.data[byte_idx] >> bit_idx) & 1;
        
        // Podwójny punkt
        SECP256k1::PointDoubleJacobian(xj, yj, zj);
        
        if (bit) {
            SECP256k1::uint256_t tx, ty, tz;
            tx = SECP256k1::Gx;
            ty = SECP256k1::Gy;
            tz.data[0] = 1;
            SECP256k1::PointAddJacobian(xj, yj, zj, tx, ty, tz, xj, yj, zj);
        }
    }
    
    // Konwersja z Jacobian
    SECP256k1::FromJacobian(xj, yj, zj, x, y);
    
    // Zapisz wyniki
    for (int i = 0; i < 8; ++i) {
        d_results[idx * 16 + i] = x.data[i];
        d_results[idx * 16 + 8 + i] = y.data[i];
    }
}

void SECP256k1::CudaMultiply(const uint256_t* keys, uint256_t* results, size_t count) {
    // Alokacja pamięci GPU
    uint32_t *d_gx, *d_gy, *d_results;
    uint256_t *d_keys;
    
    cudaMalloc(&d_gx, 8 * sizeof(uint32_t));
    cudaMalloc(&d_gy, 8 * sizeof(uint32_t));
    cudaMalloc(&d_keys, count * sizeof(uint256_t));
    cudaMalloc(&d_results, count * 16 * sizeof(uint32_t));
    
    // Kopiowanie danych
    cudaMemcpy(d_gx, Gx.data, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gy, Gy.data, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys, keys, count * sizeof(uint256_t), cudaMemcpyHostToDevice);
    
    // Konfiguracja kernela
    const int blockSize = 256;
    const int gridSize = (count + blockSize - 1) / blockSize;
    
    CudaMultiplyKernel<<<gridSize, blockSize>>>(d_gx, d_gy, d_keys, d_results, count);
    
    // Pobieranie wyników
    cudaMemcpy(results, d_results, count * 16 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Zwolnienie pamięci
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_keys);
    cudaFree(d_results);
}

#endif
