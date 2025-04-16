#include "GPUEngine.h"
#include "SECP256k1.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(err) do { \
    if(err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)) + " at line " + std::to_string(__LINE__)); \
    } \
} while(0)

#ifdef USE_CUDA

// Stałe krzywej w pamięci stałej GPU
__constant__ SECP256k1::uint256_t d_gx;
__constant__ SECP256k1::uint256_t d_gy;

//================================================
// Implementacja operacji na punktach (CUDA)
//================================================

__device__ void ModAdd(SECP256k1::uint256_t& a, const SECP256k1::uint256_t& b) {
    uint32_t carry = 0;
    for(int i = 0; i < 8; ++i) {
        uint64_t sum = static_cast<uint64_t>(a.data[i]) + 
                      static_cast<uint64_t>(b.data[i]) + carry;
        a.data[i] = static_cast<uint32_t>(sum);
        carry = sum >> 32;
    }
}

__device__ void ModSub(SECP256k1::uint256_t& a, const SECP256k1::uint256_t& b) {
    uint32_t borrow = 0;
    for(int i = 0; i < 8; ++i) {
        uint64_t diff = static_cast<uint64_t>(a.data[i]) - 
                       static_cast<uint64_t>(b.data[i]) - borrow;
        borrow = (diff >> 63) & 1;
        a.data[i] = static_cast<uint32_t>(diff);
    }
}

__device__ void PointDoubleJacobian(SECP256k1::uint256_t& x, SECP256k1::uint256_t& y, SECP256k1::uint256_t& z) {
    SECP256k1::uint256_t A, B, C, D, E, F;
    
    // A = Y^2
    SECP256k1::ModMul(y, y, A);
    
    // B = X*A
    SECP256k1::ModMul(x, A, B);
    
    // C = A^2
    SECP256k1::ModMul(A, A, C);
    
    // D = 8*Y^4
    SECP256k1::ModAdd(z, z);
    SECP256k1::ModMul(z, z, D);
    SECP256k1::ModMul(D, C, D);
    
    // E = 3X^2 - 4B
    SECP256k1::ModMul(x, x, E);
    SECP256k1::ModAdd(E, E);
    SECP256k1::ModAdd(E, E);
    SECP256k1::ModSub(E, B);
    SECP256k1::ModSub(E, B);
    
    // Nowe X
    SECP256k1::ModMul(E, E, x);
    SECP256k1::ModSub(x, B);
    SECP256k1::ModSub(x, B);
    
    // Nowe Y
    SECP256k1::ModSub(B, x, F);
    SECP256k1::ModMul(E, F, y);
    SECP256k1::ModSub(y, D);
    
    // Nowe Z
    SECP256k1::ModMul(z, A, z);
    SECP256k1::ModAdd(z, z);
}

__device__ void PointAddJacobian(SECP256k1::uint256_t& x1, SECP256k1::uint256_t& y1, SECP256k1::uint256_t& z1,
                                const SECP256k1::uint256_t& x2, const SECP256k1::uint256_t& y2) {
    SECP256k1::uint256_t z1_sq, z2_sq, u1, u2, s1, s2;
    
    // Obliczenia pośrednie
    SECP256k1::ModMul(z1, z1, z1_sq);
    SECP256k1::ModMul(z1_sq, z1, z1_sq);
    SECP256k1::ModMul(x2, z1_sq, u2);
    SECP256k1::ModMul(y2, z1_sq, s2);
    SECP256k1::ModMul(s2, z1, s2);
    
    SECP256k1::uint256_t h, r;
    SECP256k1::ModSub(u2, x1, h);
    SECP256k1::ModSub(s2, y1, r);
    
    // Obliczenia wynikowe
    SECP256k1::uint256_t h_sq, h_cu;
    SECP256k1::ModMul(h, h, h_sq);
    SECP256k1::ModMul(h_sq, h, h_cu);
    
    SECP256k1::ModMul(r, r, x1);
    SECP256k1::ModSub(x1, h_cu);
    SECP256k1::ModSub(x1, h_sq);
    
    SECP256k1::ModMul(h_sq, x1, y1);
    SECP256k1::ModSub(y1, h_cu);
    SECP256k1::ModMul(r, y1, y1);
    
    SECP256k1::ModMul(z1, h, z1);
}

//================================================
// Kernel CUDA
//================================================

__global__ void CudaMultiplyKernel(SECP256k1::uint256_t* d_keys, SECP256k1::uint256_t* d_results, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count) return;

    SECP256k1::uint256_t key = d_keys[idx];
    
    // Inicjalizacja punktu
    SECP256k1::uint256_t x = d_gx;
    SECP256k1::uint256_t y = d_gy;
    SECP256k1::uint256_t z = {1};
    
    // Montgomery Ladder
    for(int i = 255; i >= 0; --i) {
        int byte = i / 32;
        int bit = i % 32;
        bool key_bit = (key.data[byte] >> bit) & 1;
        
        PointDoubleJacobian(x, y, z);
        
        if(key_bit) {
            PointAddJacobian(x, y, z, d_gx, d_gy);
        }
    }
    
    // Konwersja do współrzędnych afinicznych
    SECP256k1::uint256_t z_inv;
    SECP256k1::ModInverse(z, z_inv);
    SECP256k1::ModMul(x, z_inv, d_results[idx]);
}

//================================================
// Implementacja klasy GPUEngine
//================================================

class GPUEngine::Impl {
public:
    Impl(int deviceId) : deviceId_(deviceId) {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        CUDA_CHECK(cudaMalloc(&d_keys_, sizeof(SECP256k1::uint256_t) * MAX_BATCH_SIZE));
        CUDA_CHECK(cudaMalloc(&d_results_, sizeof(SECP256k1::uint256_t) * MAX_BATCH_SIZE));
        
        // Skopiuj stałe krzywej do GPU
        CUDA_CHECK(cudaMemcpyToSymbol(d_gx, &SECP256k1::Gx, sizeof(SECP256k1::uint256_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_gy, &SECP256k1::Gy, sizeof(SECP256k1::uint256_t)));
    }

    ~Impl() {
        CUDA_CHECK(cudaFree(d_keys_));
        CUDA_CHECK(cudaFree(d_results_));
    }

    void SetKeys(const std::vector<SECP256k1::uint256_t>& keys) {
        keys_ = keys;
        CUDA_CHECK(cudaMemcpy(d_keys_, keys.data(), keys.size() * sizeof(SECP256k1::uint256_t), cudaMemcpyHostToDevice));
    }

    void Compute() {
        const int blockSize = 256;
        const int gridSize = (keys_.size() + blockSize - 1) / blockSize;
        
        CudaMultiplyKernel<<<gridSize, blockSize>>>(d_keys_, d_results_, keys_.size());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<SECP256k1::uint256_t> GetResults() const {
        std::vector<SECP256k1::uint256_t> results(keys_.size());
        CUDA_CHECK(cudaMemcpy(results.data(), d_results_, keys_.size() * sizeof(SECP256k1::uint256_t), cudaMemcpyDeviceToHost));
        return results;
    }

private:
    static constexpr size_t MAX_BATCH_SIZE = 1'000'000;
    int deviceId_;
    SECP256k1::uint256_t* d_keys_ = nullptr;
    SECP256k1::uint256_t* d_results_ = nullptr;
    std::vector<SECP256k1::uint256_t> keys_;
};

#else 

// Implementacja CPU

class GPUEngine::Impl {
public:
    Impl(int deviceId) {
        // Inicjalizacja środowiska CPU
        #ifdef _OPENMP
            omp_set_num_threads(omp_get_max_threads());
        #endif
    }

    void SetKeys(const std::vector<SECP256k1::uint256_t>& keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        keys_ = keys;
        results_.resize(keys_.size());
    }

    void Compute() {
        #pragma omp parallel for
        for(size_t i = 0; i < keys_.size(); ++i) {
            try {
                SECP256k1::Multiply(keys_[i], results_[i].x, results_[i].y);
            } catch(const std::exception& e) {
                #pragma omp critical
                throw std::runtime_error("CPU computation error: " + std::string(e.what()));
            }
        }
    }

    std::vector<SECP256k1::uint256_t> GetResults() const {
        std::vector<SECP256k1::uint256_t> output;
        output.reserve(results_.size());
        
        for(const auto& res : results_) {
            SECP256k1::uint256_t combined;
            std::copy(res.x.data, res.x.data + 8, combined.data);
            std::copy(res.y.data, res.y.data + 8, combined.data + 8);
            output.push_back(combined);
        }
        
        return output;
    }

private:
    struct KeyResult {
        SECP256k1::uint256_t x;
        SECP256k1::uint256_t y;
    };

    std::vector<SECP256k1::uint256_t> keys_;
    std::vector<KeyResult> results_;
    mutable std::mutex mutex_;
};

#endif

//================================================
// Interfejs publiczny
//================================================
GPUEngine::GPUEngine(int deviceId) : impl(new Impl(deviceId)) {}
GPUEngine::~GPUEngine() = default;

void GPUEngine::SetKeys(const std::vector<SECP256k1::uint256_t>& keys) { impl->SetKeys(keys); }
void GPUEngine::Compute() { impl->Compute(); }
std::vector<SECP256k1::uint256_t> GPUEngine::GetResults() const { return impl->GetResults(); }
