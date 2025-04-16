#ifndef SECP256K1_H
#define SECP256K1_H

#include <cstdint>

class SECP256k1 {
public:
    struct uint256_t {
        uint32_t data[8];
    };

    static void Init();
    static void Multiply(const uint256_t& k, uint32_t* x, uint32_t* y);
    
    #ifdef USE_CUDA
    static void CudaMultiply(const uint256_t* keys, uint32_t* results, int count);
    #endif
};

#endif