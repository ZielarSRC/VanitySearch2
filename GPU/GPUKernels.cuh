// GPUKernels.cuh
#pragma once
#include <cstdint>

#define _64K 65536
#define ITEM_SIZE 40
#define HASH_SIZE 20

__constant__ uint32_t c_constants[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// Funkcje pomocnicze
__device__ bool is_infinity(const uint64_t* X, const uint64_t* Y, const uint64_t* Z) {
    return (Z[0] | Z[1] | Z[2] | Z[3]) == 0;
}

__device__ void set_infinity(uint64_t* X, uint64_t* Y, uint64_t* Z) {
    for (int i = 0; i < 4; i++) {
        X[i] = Y[i] = 0;
        Z[i] = 1;
    }
}

__device__ bool modular_eq(const uint64_t* a, const uint64_t* b, const uint64_t* p) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

__device__ bool modular_iszero(const uint64_t* a) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

__device__ bool modular_geq(const uint64_t* a, const uint64_t* b) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

__device__ void modular_add(const uint64_t* a, const uint64_t* b, uint64_t* res, const uint64_t* p) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t s = a[i] + b[i] + carry;
        carry = (s < a[i]) ? 1 : (s == a[i] ? carry : 0);
        res[i] = s;
    }
    if (carry || modular_geq(res, p)) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t d = res[i] - p[i] - borrow;
            borrow = (res[i] < p[i] + borrow) ? 1 : 0;
            res[i] = d;
        }
    }
}

__device__ void modular_sub(const uint64_t* a, const uint64_t* b, uint64_t* res, const uint64_t* p) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t d = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i] + borrow) ? 1 : 0;
        res[i] = d;
    }
    if (borrow) {
        modular_add(res, p, res, p);
    }
}

__device__ void modular_mul(const uint64_t* a, const uint64_t* b, uint64_t* res, const uint64_t* p) {
    uint64_t product[8] = {0};
    
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t tmp = (__uint128_t)a[i] * b[j] + product[i+j] + carry;
            product[i+j] = (uint64_t)tmp;
            carry = (uint64_t)(tmp >> 64);
        }
        product[i+4] = carry;
    }
    
    uint64_t tmp[4];
    for (int i = 7; i >= 4; i--) {
        if (product[i] == 0) continue;
        
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t t = (__uint128_t)product[i] * p[j] + carry;
            tmp[j] = (uint64_t)t;
            carry = (uint64_t)(t >> 64);
        }
        
        uint64_t borrow = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t s = product[i-4+j] - tmp[j] - borrow;
            borrow = (product[i-4+j] < tmp[j] + borrow) ? 1 : 0;
            product[i-4+j] = s;
        }
        product[i] -= borrow;
    }
    
    for (int i = 0; i < 4; i++) {
        res[i] = product[i];
    }
    
    if (product[3] >> 63) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t s = res[i] + p[i] + borrow;
            borrow = (s < res[i]) ? 1 : (s == res[i] ? borrow : 0);
            res[i] = s;
        }
    }
}

__device__ void modular_sqr(const uint64_t* a, uint64_t* res, const uint64_t* p) {
    modular_mul(a, a, res, p);
}

__device__ void point_copy(uint64_t* X1, uint64_t* Y1, uint64_t* Z1, 
                          const uint64_t* X2, const uint64_t* Y2) {
    for (int i = 0; i < 4; i++) {
        X1[i] = X2[i];
        Y1[i] = Y2[i];
        Z1[i] = (i == 0) ? 1 : 0;
    }
}

__device__ void point_double(uint64_t* X, uint64_t* Y, uint64_t* Z,
                           const uint64_t* p, const uint64_t* a) {
    if (is_infinity(X, Y, Z)) return;
    if (modular_iszero(Y)) {
        set_infinity(X, Y, Z);
        return;
    }

    uint64_t A[4], B[4], C[4], D[4], E[4], F[4], tmp[4];
    
    modular_sqr(X, A, p);
    modular_sqr(Y, B, p);
    modular_sqr(B, C, p);
    
    modular_add(X, B, D, p);
    modular_sqr(D, D, p);
    modular_sub(D, A, D, p);
    modular_sub(D, C, D, p);
    modular_add(D, D, D, p);
    
    modular_add(A, A, E, p);
    modular_add(E, A, E, p);
    if (!modular_iszero(a)) {
        modular_sqr(Z, tmp, p);
        modular_sqr(tmp, tmp, p);
        modular_mul(a, tmp, tmp, p);
        modular_add(E, tmp, E, p);
    }
    
    modular_sqr(E, F, p);
    
    modular_sub(F, D, X, p);
    modular_sub(X, D, X, p);
    
    modular_sub(D, X, Y, p);
    modular_mul(E, Y, Y, p);
    modular_add(C, C, tmp, p);
    modular_add(tmp, tmp, tmp, p);
    modular_sub(Y, tmp, Y, p);
    
    modular_mul(Y, Z, Z, p);
    modular_add(Z, Z, Z, p);
}

__device__ void point_add(uint64_t* X1, uint64_t* Y1, uint64_t* Z1,
                         const uint64_t* X2, const uint64_t* Y2,
                         const uint64_t* p, const uint64_t* a) {
    if (is_infinity(X1, Y1, Z1)) {
        point_copy(X1, Y1, Z1, X2, Y2);
        return;
    }
    if (is_infinity(X2, Y2)) {
        return;
    }

    uint64_t Z1Z1[4], Z2Z2[4], U1[4], U2[4], S1[4], S2[4], H[4], I[4], J[4], r[4], V[4], tmp[4];
    
    modular_sqr(Z1, Z1Z1, p);
    modular_sqr(Z2, Z2Z2, p);
    modular_mul(X1, Z2Z2, U1, p);
    modular_mul(X2, Z1Z1, U2, p);
    
    modular_mul(Z2, Z2Z2, tmp, p);
    modular_mul(Y1, tmp, S1, p);
    modular_mul(Z1, Z1Z1, tmp, p);
    modular_mul(Y2, tmp, S2, p);
    
    if (modular_eq(U1, U2, p)) {
        if (modular_eq(S1, S2, p)) {
            point_double(X1, Y1, Z1, p, a);
            return;
        } else {
            set_infinity(X1, Y1, Z1);
            return;
        }
    }
    
    modular_sub(U2, U1, H, p);
    modular_add(H, H, tmp, p);
    modular_sqr(tmp, I, p);
    modular_mul(H, I, J, p);
    modular_sub(S2, S1, r, p);
    modular_add(r, r, r, p);
    modular_mul(U1, I, V, p);
    
    modular_sqr(r, X1, p);
    modular_sub(X1, J, X1, p);
    modular_sub(X1, V, X1, p);
    modular_sub(X1, V, X1, p);
    
    modular_sub(V, X1, Y1, p);
    modular_mul(r, Y1, Y1, p);
    modular_mul(S1, J, tmp, p);
    modular_add(tmp, tmp, tmp, p);
    modular_sub(Y1, tmp, Y1, p);
    
    modular_add(Z1, Z2, Z1, p);
    modular_sqr(Z1, Z1, p);
    modular_sub(Z1, Z1Z1, Z1, p);
    modular_sub(Z1, Z2Z2, Z1, p);
    modular_mul(Z1, H, Z1, p);
}

__device__ void generate_pubkey(const uint64_t* privkey, uint64_t* pubx, uint64_t* puby) {
    const uint64_t Gx[4] = {0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB};
    const uint64_t Gy[4] = {0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448};
    const uint64_t p[4] = {0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
    const uint64_t a[4] = {0, 0, 0, 0};
    const uint64_t b[4] = {7, 0, 0, 0};

    uint64_t k[4];
    for (int i = 0; i < 4; i++) k[i] = privkey[i];
    
    uint64_t Qx[4], Qy[4], Qz[4] = {1,0,0,0};
    uint64_t tmp[4];
    
    point_copy(Qx, Gx, Qy, Gy, Qz);
    
    for (int i = 255; i >= 0; i--) {
        point_double(Qx, Qy, Qz, p, a);
        if ((k[i/64] >> (i%64)) & 1) {
            point_add(Qx, Qy, Qz, Gx, Gy, p, a);
        }
    }
    
    modular_inv(Qz, tmp, p);
    modular_mul(Qx, tmp, pubx, p);
    modular_mul(Qy, tmp, puby, p);
}

__device__ void get_hash160(const uint64_t* x, const uint64_t* y, bool compressed, uint8_t* hash) {
    uint8_t pubkey[65];
    uint8_t sha256[32];
    
    if (compressed) {
        pubkey[0] = (y[0] & 1) ? 0x03 : 0x02;
        for (int i = 0; i < 4; i++) {
            *((uint64_t*)(pubkey + 1 + i*8)) = x[i];
        }
        sha256_full(pubkey, 33, sha256);
    } else {
        pubkey[0] = 0x04;
        for (int i = 0; i < 4; i++) {
            *((uint64_t*)(pubkey + 1 + i*8)) = x[i];
            *((uint64_t*)(pubkey + 33 + i*8)) = y[i];
        }
        sha256_full(pubkey, 65, sha256);
    }
    
    ripemd160_full(sha256, 32, hash);
}

template <bool UseSharedMem>
__global__ void optimized_comp_keys(
    uint32_t searchMode,
    const uint16_t* __restrict__ prefixes,
    const uint32_t* __restrict__ prefixLookup,
    const uint64_t* __restrict__ keys,
    uint32_t maxFound,
    uint32_t* output
) {
    extern __shared__ uint8_t sharedMem[];
    uint16_t* sharedPrefixes = reinterpret_cast<uint16_t*>(sharedMem);
    
    if (UseSharedMem) {
        for (int i = threadIdx.x; i < _64K; i += blockDim.x) {
            sharedPrefixes[i] = prefixes[i];
        }
        __syncthreads();
    }
    
    const uint16_t* activePrefixes = UseSharedMem ? sharedPrefixes : prefixes;
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t* key = &keys[threadId * 8];
    
    uint64_t pubx[4], puby[4];
    generate_pubkey(key, pubx, puby);
    
    uint8_t hash[HASH_SIZE];
    get_hash160(pubx, puby, false, hash);
    
    if (check_hash(hash, activePrefixes, prefixLookup)) {
        uint32_t pos = atomicAdd(output, 1);
        if (pos < maxFound) {
            uint32_t* out = output + 1 + pos * (ITEM_SIZE / 4);
            out[0] = threadId;
            *reinterpret_cast<int16_t*>(&out[1]) = 0;
            *reinterpret_cast<int16_t*>(&out[1] + 1) = 0;
            memcpy(out + 2, hash, HASH_SIZE);
        }
    }
    
    if (searchMode == static_cast<uint32_t>(SearchMode::Both)) {
        get_hash160(pubx, puby, true, hash);
        if (check_hash(hash, activePrefixes, prefixLookup)) {
            uint32_t pos = atomicAdd(output, 1);
            if (pos < maxFound) {
                uint32_t* out = output + 1 + pos * (ITEM_SIZE / 4);
                out[0] = threadId;
                *reinterpret_cast<int16_t*>(&out[1]) = 0x8000;
                *reinterpret_cast<int16_t*>(&out[1] + 1) = 0;
                memcpy(out + 2, hash, HASH_SIZE);
            }
        }
    }
}
