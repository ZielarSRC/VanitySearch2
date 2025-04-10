/*
 * Modernized GPUMath for VanitySearch2
 * CUDA 12+ Optimized Elliptic Curve Operations
 */

#include "GPUMath.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_arithmetic.h>
#include <array>
#include <limits>

namespace cg = cooperative_groups;

// Constants optimized for Ampere/Ada Lovelace
__constant__ uint64_t _P[4];
__constant__ uint64_t _R[4];
__constant__ uint64_t _R2[4];
__constant__ uint64_t _3[4];
__constant__ uint64_t _beta[4];
__constant__ uint64_t _beta2[4];

// Shared memory optimization
struct SMem {
    __device__ inline static uint64_t* reduceTemp() {
        __shared__ uint64_t tmp[8];
        return tmp;
    }
    
    __device__ inline static uint64_t* multTemp() {
        __shared__ uint64_t tmp[16];
        return tmp;
    }
};

// Fast 256-bit modular reduction
__device__ void _ModReduce(uint64_t* r, const uint64_t* t) {
    auto block = cg::this_thread_block();
    uint64_t* tmp = SMem::reduceTemp();
    
    if (threadIdx.x < 8) {
        tmp[threadIdx.x] = t[threadIdx.x];
    }
    block.sync();

    // Reduction using schoolbook method
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        uint64_t word = tmp[i + 4];
        
        // Add with carry
        tmp[i] += word;
        carry = tmp[i] < word;
        
        // Propagate carry
        for (int j = i + 1; j < 4 && carry; j++) {
            uint64_t old = tmp[j];
            tmp[j] += carry;
            carry = tmp[j] < old;
        }
    }
    block.sync();

    if (threadIdx.x < 4) {
        r[threadIdx.x] = tmp[threadIdx.x];
    }
    
    // Final conditional subtraction
    if (threadIdx.x == 0) {
        bool subtract = (tmp[3] > _P[3]) ||
                       ((tmp[3] == _P[3]) && (tmp[2] > _P[2])) ||
                       ((tmp[3] == _P[3]) && (tmp[2] == _P[2]) && (tmp[1] > _P[1])) ||
                       ((tmp[3] == _P[3]) && (tmp[2] == _P[2]) && (tmp[1] == _P[1]) && (tmp[0] >= _P[0]));
        
        if (subtract) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                r[i] = tmp[i] - _P[i];
            }
        }
    }
}

// Optimized 256-bit multiplication
__device__ void _ModMult(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    auto block = cg::this_thread_block();
    uint64_t* tmp = SMem::multTemp();
    
    // Schoolbook multiplication
    if (threadIdx.x < 16) {
        tmp[threadIdx.x] = 0;
    }
    block.sync();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        uint64_t word = a[i];
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(word), "l"(b[j]));
            asm("mad.lo.cc.u64 %0, %1, %2, %3;" : "=l"(lo) : "l"(word), "l"(b[j]), "l"(tmp[i + j]));
            asm("madc.hi.u64 %0, %1, %2, %3;" : "=l"(carry) : "l"(word), "l"(b[j]), "l"(hi));
            
            tmp[i + j] = lo;
            if (j < 3) tmp[i + j + 1] += carry;
        }
    }
    block.sync();

    // Reduction
    _ModReduce(r, tmp);
}

// Optimized modular squaring
__device__ void _ModSqr(uint64_t* r, const uint64_t* a) {
    _ModMult(r, a, a);
}

// Modular inversion using Fermat's Little Theorem
__device__ void _ModInv(uint64_t* r, const uint64_t* a) {
    uint64_t t0[4], t1[4], t2[4];
    
    // a^(p-2) mod p
    Load256(t0, a);
    Load256(t1, _R2);  // Convert to Montgomery form
    
    // Exponent p-2 in binary: 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E
    #pragma unroll
    for (int i = 254; i >= 0; i--) {
        _ModSqr(t1, t1);
        if (i == 255 || i == 32 || i == 0) {
            _ModMult(t2, t1, t0);
            if ((0xFFFFFFFFFFFFFFFEFFFFFC2EULL >> i) & 1) {
                Load256(t1, t2);
            }
        }
    }
    
    Store256(r, t1);
}

// Grouped modular inversion using Montgomery trick
__device__ void _ModInvGrouped(uint64_t dx[][4]) {
    auto block = cg::this_thread_block();
    const int groupSize = GRP_SIZE / 2 + 1;
    __shared__ uint64_t products[groupSize][4];
    __shared__ uint64_t inverse[4];
    
    // Compute product tree
    if (threadIdx.x < groupSize) {
        Load256(products[threadIdx.x], dx[threadIdx.x]);
    }
    block.sync();

    for (int stride = 1; stride < groupSize; stride *= 2) {
        if (threadIdx.x % (2 * stride) == 0 && threadIdx.x + stride < groupSize) {
            _ModMult(products[threadIdx.x], products[threadIdx.x], products[threadIdx.x + stride]);
        }
        block.sync();
    }

    // Compute inverse of final product
    if (threadIdx.x == 0) {
        _ModInv(inverse, products[0]);
    }
    block.sync();

    // Compute inverse products
    for (int stride = groupSize / 2; stride >= 1; stride /= 2) {
        if (threadIdx.x % (2 * stride) == 0 && threadIdx.x + stride < groupSize) {
            uint64_t temp[4];
            Load256(temp, products[threadIdx.x + stride]);
            
            _ModMult(products[threadIdx.x + stride], products[threadIdx.x], inverse);
            _ModMult(products[threadIdx.x], temp, inverse);
        }
        block.sync();
    }

    // Store results
    if (threadIdx.x < groupSize) {
        Load256(dx[threadIdx.x], products[threadIdx.x]);
    }
}

// Point addition in Jacobian coordinates
__device__ void _AddJac(uint64_t* x3, uint64_t* y3, uint64_t* z3,
                       const uint64_t* x1, const uint64_t* y1, const uint64_t* z1,
                       const uint64_t* x2, const uint64_t* y2) {
    uint64_t t0[4], t1[4], t2[4], t3[4], t4[4];
    
    // Z1Z1 = Z1²
    _ModSqr(t0, z1);
    
    // U2 = X2*Z1Z1
    _ModMult(t1, x2, t0);
    
    // S2 = Y2*Z1*Z1Z1
    _ModMult(t0, t0, z1);
    _ModMult(t2, y2, t0);
    
    // H = U2-X1
    ModSub256(t1, t1, x1);
    
    // HH = H²
    _ModSqr(t3, t1);
    
    // I = 4*HH
    ModAdd256(t4, t3, t3);
    ModAdd256(t4, t4, t4);
    
    // J = H*I
    _ModMult(t3, t3, t1);
    
    // r = 2*(S2-Y1)
    ModSub256(t2, t2, y1);
    ModAdd256(t2, t2, t2);
    
    // V = X1*I
    _ModMult(t1, x1, t4);
    
    // X3 = r²-J-2*V
    _ModSqr(x3, t2);
    ModSub256(x3, x3, t3);
    ModSub256(x3, x3, t1);
    ModSub256(x3, x3, t1);
    
    // Y3 = r*(V-X3)-2*Y1*J
    ModSub256(t1, t1, x3);
    _ModMult(t1, t1, t2);
    _ModMult(y3, y1, t3);
    ModAdd256(y3, y3, y3);
    ModSub256(y3, t1, y3);
    
    // Z3 = (Z1+H)²-Z1Z1-HH
    ModAdd256(z3, z1, t1);
    _ModSqr(z3, z3);
    ModSub256(z3, z3, t0);
    ModSub256(z3, z3, t3);
}

// Initialize curve parameters
void InitGPUMath(const uint64_t* P, const uint64_t* R, const uint64_t* R2, 
                const uint64_t* beta, const uint64_t* beta2) {
    uint64_t three[4] = {3, 0, 0, 0};
    
    cudaMemcpyToSymbol(_P, P, 32);
    cudaMemcpyToSymbol(_R, R, 32);
    cudaMemcpyToSymbol(_R2, R2, 32);
    cudaMemcpyToSymbol(_3, three, 32);
    cudaMemcpyToSymbol(_beta, beta, 32);
    cudaMemcpyToSymbol(_beta2, beta2, 32);
}

// Optimized modular addition
__device__ void ModAdd256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a[i] + b[i] + carry;
        carry = (sum < a[i]) || (carry && (sum == a[i]));
        r[i] = sum;
    }
    
    // Conditional subtraction
    if (carry || (r[3] > _P[3]) || 
        ((r[3] == _P[3]) && (r[2] > _P[2])) ||
        ((r[3] == _P[3]) && (r[2] == _P[2]) && (r[1] > _P[1])) ||
        ((r[3] == _P[3]) && (r[2] == _P[2]) && (r[1] == _P[1]) && (r[0] >= _P[0]))) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            r[i] -= _P[i];
        }
    }
}

// Optimized modular subtraction
__device__ void ModSub256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t diff = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i]) || (borrow && (a[i] == b[i]));
        r[i] = diff;
    }
    
    // Conditional addition
    if (borrow) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            r[i] += _P[i];
        }
    }
}

// Modular negation
__device__ void ModNeg256(uint64_t* r, const uint64_t* a) {
    if (threadIdx.x < 4) {
        r[threadIdx.x] = _P[threadIdx.x] - a[threadIdx.x];
        if (threadIdx.x == 0 && r[0] == 0) {
            // Handle special case
            #pragma unroll
            for (int i = 1; i < 4; i++) {
                r[i] = _P[i] - a[i];
            }
        }
    }
}

// Load 256-bit value with memory alignment
__device__ void Load256(uint64_t* r, const uint64_t* a) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = a[i];
    }
}

// Store 256-bit value with memory alignment
__device__ void Load256A(uint64_t* r, const uint64_t* a) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = a[i];
    }
}

// Store 256-bit value with memory alignment
__device__ void Store256(uint64_t* r, const uint64_t* a) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = a[i];
    }
}

// Store 256-bit value with memory alignment
__device__ void Store256A(uint64_t* r, const uint64_t* a) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = a[i];
    }
}