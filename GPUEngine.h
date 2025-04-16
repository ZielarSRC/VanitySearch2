#ifndef GPUENGINE_H
#define GPUENGINE_H

#include "SECP256k1.h"
#include <memory>
#include <vector>

class GPUEngine {
public:
    explicit GPUEngine(int deviceId = 0);
    ~GPUEngine();

    void SetKeys(const std::vector<SECP256k1::uint256_t>& keys);
    void Compute();
    std::vector<SECP256k1::uint256_t> GetResults() const;

    // Usuwanie konstruktorów kopiujących
    GPUEngine(const GPUEngine&) = delete;
    GPUEngine& operator=(const GPUEngine&) = delete;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

#endif
