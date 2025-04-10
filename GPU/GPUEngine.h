#pragma once
#include <cuda_runtime.h>
#include "GPUGroup.h"
#include "GPUHash.h"
#include "GPUMath.h"

class GPUEngine {
public:
    // Konstruktor z obsługą wielu GPU
    explicit GPUEngine(int deviceID = 0);

    // Nowoczesne zarządzanie pamięcią (CUDA 12+)
    template<typename T>
    T* allocateDeviceMemory(size_t count, cudaStream_t stream = 0);

    template<typename T>
    void freeDeviceMemory(T* ptr, cudaStream_t stream = 0);

    // Generowanie kluczy (zoptymalizowane pod Ampere)
    __global__ void generateKeys(uint64_t* privateKeys, uint64_t* publicKeys, size_t count);

    // Wyszukiwanie adresów (z wildcard)
    __host__ __device__ bool matchAddress(const char* address, const char* pattern);

    // Synchronizacja strumienia
    void synchronizeStream(cudaStream_t stream = 0);

private:
    int deviceID_;
    cudaDeviceProp deviceProps_;
};
