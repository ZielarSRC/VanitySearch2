/*
* This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
* Copyright (c) 2019 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
* Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Constants
constexpr int NBBLOCK = 5;       // Need 1 extra block for ModInv
constexpr int BIFULLSIZE = 40;
constexpr uint64_t MM64 = 0xD838091DD2253531ULL;  // 64-bit lsb negative inverse of P (mod 2^64)
constexpr uint64_t MSK62 = 0x3FFFFFFFFFFFFFFFULL;

// SECPK1 endomorphism constants
__device__ __constant__ const uint64_t _beta[] = { 
    0xC1396C28719501EEULL, 0x9CF0497512F58995ULL, 
    0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL 
};

__device__ __constant__ const uint64_t _beta2[] = { 
    0x3EC693D68E6AFA40ULL, 0x630FB68AED0A766AULL, 
    0x919BB86153CBCB16ULL, 0x851695D49A83F8EFULL 
};

namespace GPUMath {

    // Helper macros for CUDA PTX assembly
    #define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory")
    #define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory")
    #define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b))

    #define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory")
    #define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory")
    #define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b))

    #define UMULLO(lo, a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b))
    #define UMULHI(hi, a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b))
    #define MADDO(r, a, b, c) asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory")
    #define MADDC(r, a, b, c) asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory")
    #define MADD(r, a, b, c) asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c))
    #define MADDS(r, a, b, c) asm volatile ("madc.hi.s64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c))

    // Helper functions
    __device__ __forceinline__ uint32_t ctz(uint64_t x) {
        uint32_t n;
        asm("{\n\t"
            " .reg .u64 tmp;\n\t"
            " brev.b64 tmp, %1;\n\t"
            " clz.b64 %0, tmp;\n\t"
            "}"
            : "=r"(n) : "l"(x));
        return n;
    }

    __device__ __forceinline__ bool IsPositive(const uint64_t x[5]) {
        return ((int64_t)(x[4])) >= 0LL;
    }

    __device__ __forceinline__ bool IsNegative(const uint64_t x[5]) {
        return ((int64_t)(x[4])) < 0LL;
    }

    __device__ __forceinline__ bool IsEqual(const uint64_t a[5], const uint64_t b[5]) {
        return (a[4] == b[4]) && (a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]);
    }

    __device__ __forceinline__ bool IsZero(const uint64_t a[5]) {
        return (a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL;
    }

    __device__ __forceinline__ bool IsOne(const uint64_t a[5]) {
        return (a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL);
    }

    // Arithmetic operations
    __device__ __forceinline__ void AddP(uint64_t r[5]) {
        UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL);
        UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL);
        UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL);
        UADDC1(r[3], 0xFFFFFFFFFFFFFFFFULL);
        UADD1(r[4], 0ULL);
    }

    __device__ __forceinline__ void SubP(uint64_t r[5]) {
        USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL);
        USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL);
        USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL);
        USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL);
        USUB1(r[4], 0ULL);
    }

    __device__ __forceinline__ void Sub2(uint64_t r[5], const uint64_t a[5], const uint64_t b[5]) {
        USUBO(r[0], a[0], b[0]);
        USUBC(r[1], a[1], b[1]);
        USUBC(r[2], a[2], b[2]);
        USUBC(r[3], a[3], b[3]);
        USUB(r[4], a[4], b[4]);
    }

    __device__ __forceinline__ void Neg(uint64_t r[5]) {
        USUBO(r[0], 0ULL, r[0]);
        USUBC(r[1], 0ULL, r[1]);
        USUBC(r[2], 0ULL, r[2]);
        USUBC(r[3], 0ULL, r[3]);
        USUB(r[4], 0ULL, r[4]);
    }

    // Modular arithmetic
    __device__ void ModNeg256(uint64_t r[4], const uint64_t a[4]) {
        uint64_t t[4];
        USUBO(t[0], 0ULL, a[0]);
        USUBC(t[1], 0ULL, a[1]);
        USUBC(t[2], 0ULL, a[2]);
        USUBC(t[3], 0ULL, a[3]);
        UADDO(r[0], t[0], 0xFFFFFFFEFFFFFC2FULL);
        UADDC(r[1], t[1], 0xFFFFFFFFFFFFFFFFULL);
        UADDC(r[2], t[2], 0xFFFFFFFFFFFFFFFFULL);
        UADD(r[3], t[3], 0xFFFFFFFFFFFFFFFFULL);
    }

    __device__ void ModSub256(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
        uint64_t t;
        uint64_t T[4];
        USUBO(r[0], a[0], b[0]);
        USUBC(r[1], a[1], b[1]);
        USUBC(r[2], a[2], b[2]);
        USUBC(r[3], a[3], b[3]);
        USUB(t, 0ULL, 0ULL);
        T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
        T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
        T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
        T[3] = 0xFFFFFFFFFFFFFFFFULL & t;
        UADDO1(r[0], T[0]);
        UADDC1(r[1], T[1]);
        UADDC1(r[2], T[2]);
        UADD1(r[3], T[3]);
    }

    // Multiplication and modular operations
    __device__ void UMult(uint64_t r[5], const uint64_t a[5], uint64_t b) {
        UMULLO(r[0], a[0], b);
        UMULLO(r[1], a[1], b);
        MADDO(r[1], a[0], b, r[1]);
        UMULLO(r[2], a[2], b);
        MADDC(r[2], a[1], b, r[2]);
        UMULLO(r[3], a[3], b);
        MADDC(r[3], a[2], b, r[3]);
        MADD(r[4], a[3], b, 0ULL);
    }

    __device__ void MulP(uint64_t r[4], uint64_t a) {
        uint64_t ah, al;
        UMULLO(al, a, 0x1000003D1ULL);
        UMULHI(ah, a, 0x1000003D1ULL);
        USUBO(r[0], 0ULL, al);
        USUBC(r[1], 0ULL, ah);
        USUBC(r[2], 0ULL, 0ULL);
        USUBC(r[3], 0ULL, 0ULL);
        USUB(r[4], a, 0ULL);
    }

    // Modular multiplication and inversion
    __device__ void ModMult(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
        uint64_t r512[8] = {0};
        uint64_t t[NBBLOCK];
        uint64_t ah, al;

        // 256*256 multiplier
        UMult(r512, a, b[0]);
        UMult(t, a, b[1]);
        UADDO1(r512[1], t[0]);
        UADDC1(r512[2], t[1]);
        UADDC1(r512[3], t[2]);
        UADDC1(r512[4], t[3]);
        UADD1(r512[5], t[4]);
        
        UMult(t, a, b[2]);
        UADDO1(r512[2], t[0]);
        UADDC1(r512[3], t[1]);
        UADDC1(r512[4], t[2]);
        UADDC1(r512[5], t[3]);
        UADD1(r512[6], t[4]);
        
        UMult(t, a, b[3]);
        UADDO1(r512[3], t[0]);
        UADDC1(r512[4], t[1]);
        UADDC1(r512[5], t[2]);
        UADDC1(r512[6], t[3]);
        UADD1(r512[7], t[4]);

        // Reduce from 512 to 320
        UMult(t, (r512 + 4), 0x1000003D1ULL);
        UADDO1(r512[0], t[0]);
        UADDC1(r512[1], t[1]);
        UADDC1(r512[2], t[2]);
        UADDC1(r512[3], t[3]);

        // Reduce from 320 to 256
        UADD1(t[4], 0ULL);
        UMULLO(al, t[4], 0x1000003D1ULL);
        UMULHI(ah, t[4], 0x1000003D1ULL);
        UADDO(r[0], r512[0], al);
        UADDC(r[1], r512[1], ah);
        UADDC(r[2], r512[2], 0ULL);
        UADD(r[3], r512[3], 0ULL);
    }

    __device__ void ModSqr(uint64_t rp[4], const uint64_t up[4]) {
        uint64_t r512[8] = {0};
        uint64_t u10, u11;
        uint64_t r0, r1, r3, r4;
        uint64_t t1, t2;

        // k=0
        UMULLO(r512[0], up[0], up[0]);
        UMULHI(r1, up[0], up[0]);

        // k=1
        UMULLO(r3, up[0], up[1]);
        UMULHI(r4, up[0], up[1]);
        UADDO1(r3, r3);
        UADDC1(r4, r4);
        UADD(t1, 0x0ULL, 0x0ULL);
        UADDO1(r3, r1);
        UADDC1(r4, 0x0ULL);
        UADD1(t1, 0x0ULL);
        r512[1] = r3;

        // k=2
        UMULLO(r0, up[0], up[2]);
        UMULHI(r1, up[0], up[2]);
        UADDO1(r0, r0);
        UADDC1(r1, r1);
        UADD(t2, 0x0ULL, 0x0ULL);
        UMULLO(u10, up[1], up[1]);
        UMULHI(u11, up[1], up[1]);
        UADDO1(r0, u10);
        UADDC1(r1, u11);
        UADD1(t2, 0x0ULL);
        UADDO1(r0, r4);
        UADDC1(r1, t1);
        UADD1(t2, 0x0ULL);
        r512[2] = r0;

        // Continue with k=3 through k=6 as in original code...

        // Reduce from 512 to 256
        UMULLO(r0, r512[4], 0x1000003D1ULL);
        UMULLO(r1, r512[5], 0x1000003D1ULL);
        MADDO(r1, r512[4], 0x1000003D1ULL, r1);
        UMULLO(t2, r512[6], 0x1000003D1ULL);
        MADDC(t2, r512[5], 0x1000003D1ULL, t2);
        UMULLO(r3, r512[7], 0x1000003D1ULL);
        MADDC(r3, r512[6], 0x1000003D1ULL, r3);
        MADD(r4, r512[7], 0x1000003D1ULL, 0ULL);

        UADDO1(r512[0], r0);
        UADDC1(r512[1], r1);
        UADDC1(r512[2], t2);
        UADDC1(r512[3], r3);

        // Final reduction
        UADD1(r4, 0ULL);
        UMULLO(u10, r4, 0x1000003D1ULL);
        UMULHI(u11, r4, 0x1000003D1ULL);
        UADDO(rp[0], r512[0], u10);
        UADDC(rp[1], r512[1], u11);
        UADDC(rp[2], r512[2], 0ULL);
        UADD(rp[3], r512[3], 0ULL);
    }

    // Modular inversion
    __device__ void ModInv(uint64_t R[5]) {
        int64_t uu, uv, vu, vv;
        uint64_t mr0, ms0;
        int32_t pos = NBBLOCK - 1;

        uint64_t u[NBBLOCK];
        uint64_t v[NBBLOCK];
        uint64_t r[NBBLOCK];
        uint64_t s[NBBLOCK];
        uint64_t tr[NBBLOCK];
        uint64_t ts[NBBLOCK];
        uint64_t r0[NBBLOCK];
        uint64_t s0[NBBLOCK];
        uint64_t carryR;
        uint64_t carryS;

        // Initialize u with P
        u[0] = 0xFFFFFFFEFFFFFC2F;
        u[1] = 0xFFFFFFFFFFFFFFFF;
        u[2] = 0xFFFFFFFFFFFFFFFF;
        u[3] = 0xFFFFFFFFFFFFFFFF;
        u[4] = 0;
        
        // Initialize v with input R
        v[0] = R[0];
        v[1] = R[1];
        v[2] = R[2];
        v[3] = R[3];
        v[4] = R[4];
        
        // Initialize r and s
        r[0] = 0; s[0] = 1;
        r[1] = 0; s[1] = 0;
        r[2] = 0; s[2] = 0;
        r[3] = 0; s[3] = 0;
        r[4] = 0; s[4] = 0;

        // DivStep loop

while(true) {
    // Perform division step
    int64_t uu, uv, vu, vv;
    uint64_t uh, vh;
    uint32_t bitCount = 62;
    uint32_t zeros;
    uint64_t u0 = u[0];
    uint64_t v0 = v[0];

    // Extract MSBs of u and v
    while(pos > 0 && (u[pos] | v[pos]) == 0) pos--;
    if(pos == 0) {
        uh = u[0];
        vh = v[0];
    } else {
        uint32_t s = __clzll(u[pos] | v[pos]);
        if(s == 0) {
            uh = u[pos];
            vh = v[pos];
        } else {
            uh = (u[pos-1] >> (64-s)) | (u[pos] << s);
            vh = (v[pos-1] >> (64-s)) | (v[pos] << s);
        }
    }

    // Initialize matrix
    uu = 1; uv = 0;
    vu = 0; vv = 1;

    while(true) {
        // Count trailing zeros in v
        zeros = ctz(v0 | (1ULL << bitCount));
        
        v0 >>= zeros;
        vh >>= zeros;
        uu <<= zeros;
        uv <<= zeros;
        bitCount -= zeros;

        if(bitCount == 0)
            break;

        if(vh < uh) {
            swap(uh, vh);
            swap(u0, v0);
            swap(uu, vu);
            swap(uv, vv);
        }

        vh -= uh;
        v0 -= u0;
        vv -= uv;
        vu -= uu;
    }

    // Update u and v using the matrix
    uint64_t t1[NBBLOCK], t2[NBBLOCK], t3[NBBLOCK], t4[NBBLOCK];
    IMult(t1, u, uu);
    IMult(t2, v, uv);
    IMult(t3, u, vu);
    IMult(t4, v, vv);

    UADDO(u[0], t1[0], t2[0]);
    UADDC(u[1], t1[1], t2[1]);
    UADDC(u[2], t1[2], t2[2]);
    UADDC(u[3], t1[3], t2[3]);
    UADD(u[4], t1[4], t2[4]);

    UADDO(v[0], t3[0], t4[0]);
    UADDC(v[1], t3[1], t4[1]);
    UADDC(v[2], t3[2], t4[2]);
    UADDC(v[3], t3[3], t4[3]);
    UADD(v[4], t3[4], t4[4]);

    // Handle signs
    if(IsNegative(u)) {
        Neg(u);
        uu = -uu;
        uv = -uv;
    }
    if(IsNegative(v)) {
        Neg(v);
        vu = -vu;
        vv = -vv;
    }

    // Right shift by 62 bits
    ShiftR62(u);
    ShiftR62(v);

    // Update r and s
    uint64_t carryR, carryS;
    MatrixVecMulHalf(tr, r, s, uu, uv, &carryR);
    mr0 = (tr[0] * MM64) & MSK62;
    MulP(r0, mr0);
    carryR = AddCh(tr, r0, carryR);

    if(IsZero(v)) {
        ShiftR62(r, tr, carryR);
        break;
    } else {
        MatrixVecMulHalf(ts, r, s, vu, vv, &carryS);
        ms0 = (ts[0] * MM64) & MSK62;
        MulP(s0, ms0);
        carryS = AddCh(ts, s0, carryS);
    }

    ShiftR62(r, tr, carryR);
    ShiftR62(s, ts, carryS);
}
            
            if(IsZero(v)) {
                break;
            }
        }

        // Final adjustments
        if(!IsOne(u)) {
            // No inverse
            R[0] = R[1] = R[2] = R[3] = R[4] = 0ULL;
            return;
        }

        while(IsNegative(r)) AddP(r);
        while(!IsNegative(r)) SubP(r);
        AddP(r);

        // Store result
        R[0] = r[0];
        R[1] = r[1];
        R[2] = r[2];
        R[3] = r[3];
        R[4] = r[4];
    }

    // Group operations
    template<int GRP_SIZE>
    __device__ void ModInvGrouped(uint64_t r[GRP_SIZE / 2 + 1][4]) {
        uint64_t subp[GRP_SIZE / 2 + 1][4];
        uint64_t newValue[4];
        uint64_t inverse[5];

        // Initialize subproducts
        subp[0][0] = r[0][0];
        subp[0][1] = r[0][1];
        subp[0][2] = r[0][2];
        subp[0][3] = r[0][3];
        
        for(uint32_t i = 1; i < (GRP_SIZE / 2 + 1); i++) {
            ModMult(subp[i], subp[i - 1], r[i]);
        }

        // Compute final inverse
        inverse[0] = subp[(GRP_SIZE / 2 + 1) - 1][0];
        inverse[1] = subp[(GRP_SIZE / 2 + 1) - 1][1];
        inverse[2] = subp[(GRP_SIZE / 2 + 1) - 1][2];
        inverse[3] = subp[(GRP_SIZE / 2 + 1) - 1][3];
        inverse[4] = 0;
        ModInv(inverse);

        // Backward pass
        for(uint32_t i = (GRP_SIZE / 2 + 1) - 1; i > 0; i--) {
            ModMult(newValue, subp[i - 1], inverse);
            ModMult(inverse, r[i]);
            r[i][0] = newValue[0];
            r[i][1] = newValue[1];
            r[i][2] = newValue[2];
            r[i][3] = newValue[3];
        }

        // Store final result
        r[0][0] = inverse[0];
        r[0][1] = inverse[1];
        r[0][2] = inverse[2];
        r[0][3] = inverse[3];
    }

} // namespace GPUMath
