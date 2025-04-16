#ifndef GPUENGINE_H
#define GPUENGINE_H

#include <vector>
#include "SECP256k1.h"

class GPUEngine {
public:
    GPUEngine(int deviceId = 0);
    ~GPUEngine();
    
    void SetKeys(const std::vector<SECP256k1::uint256_t>& keys);
    void Compute();
    std::vector<uint32_t> GetResults();

private:
    class Impl;
    Impl* impl;
};

#endif