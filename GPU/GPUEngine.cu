// GPUEngine.cu
#include "GPUEngine.h"
#include "GPUKernels.cuh"
#include <stdexcept>
#include <cstdio>
#include <cstring>

#define _64K 65536
#define ITEM_SIZE 40
#define ITEM_SIZE32 (ITEM_SIZE / 4)
#define GRP_SIZE 1024
#define STEP_SIZE 256
#define BIFULLSIZE 32

namespace GPUEngine {

namespace {
    constexpr const char* CUDA_ERROR_PREFIX = "CUDA error: ";

    void checkCudaError(cudaError_t err, const char* context) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(CUDA_ERROR_PREFIX) + context + ": " + cudaGetErrorString(err));
        }
    }

    int convertSMVerToCores(int major, int minor) {
        struct SMToCores { int version; int cores; };
        const SMToCores archCoresPerSM[] = {
            {0x20, 32}, {0x21, 48}, {0x30, 192}, {0x32, 192},
            {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128},
            {0x53, 128}, {0x60, 64}, {0x61, 128}, {0x62, 128},
            {0x70, 64}, {0x72, 64}, {0x75, 64}, {0x80, 64},
            {0x86, 128}, {-1, -1}
        };

        const int version = (major << 4) + minor;
        for (const auto& entry : archCoresPerSM) {
            if (entry.version == version) return entry.cores;
        }
        return 0;
    }
}

CudaContext::CudaContext(int device_id) : device_id_(device_id) {
    checkCudaError(cudaSetDevice(device_id_), "Set device");
    checkCudaError(cudaGetDeviceProperties(&prop_, device_id_), "Get device properties");
    checkCudaError(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1), "Set cache config");
    checkCudaError(cudaDeviceSetLimit(cudaLimitStackSize, 49152), "Set stack size");
}

CudaContext::~CudaContext() {
    if (device_id_ >= 0) cudaDeviceReset();
}

GPUMemory::GPUMemory(size_t size) { allocate(size); }
GPUMemory::~GPUMemory() { free(); }

void GPUMemory::allocate(size_t size) {
    if (ptr_) free();
    checkCudaError(cudaMalloc(&ptr_, size), "Allocate device memory");
    size_ = size;
}

void GPUMemory::free() {
    if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; size_ = 0; }
}

PinnedMemory::PinnedMemory(size_t size, unsigned int flags) { allocate(size, flags); }
PinnedMemory::~PinnedMemory() { free(); }

void PinnedMemory::allocate(size_t size, unsigned int flags) {
    if (ptr_) free();
    checkCudaError(cudaHostAlloc(&ptr_, size, flags), "Allocate pinned memory");
    size_ = size;
}

void PinnedMemory::free() {
    if (ptr_) { cudaFreeHost(ptr_); ptr_ = nullptr; size_ = 0; }
}

Engine::Engine(int threadGroups, int threadsPerGroup, int gpuId, uint32_t maxFound, bool rekey) 
    : totalThreads_(0), threadsPerGroup_(threadsPerGroup), maxFound_(maxFound), 
      rekeyEnabled_(rekey), searchMode_(SearchMode::Compressed), searchType_(SearchType::P2PKH) {
    try {
        context_ = std::make_unique<CudaContext>(gpuId);
        initialize();
    } catch (...) {
        cleanup();
        throw;
    }
}

Engine::~Engine() { cleanup(); }

void Engine::initialize() {
    const auto& props = context_->properties();
    if (threadGroups == -1) threadGroups = props.multiProcessorCount * 8;
    
    totalThreads_ = threadGroups * threadsPerGroup_;
    outputSize_ = (maxFound_ * ITEM_SIZE + 4);

    char buffer[512];
    snprintf(buffer, sizeof(buffer), "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
            context_->device_id(), props.name, props.multiProcessorCount,
            convertSMVerToCores(props.major, props.minor),
            totalThreads_ / threadsPerGroup_, threadsPerGroup_);
    deviceName_ = buffer;

    devicePrefixes_.allocate(_64K * 2);
    deviceKeys_.allocate(totalThreads_ * 32 * 2);
    deviceOutput_.allocate(outputSize_);
    hostPrefixes_.allocate(_64K * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    hostKeys_.allocate(totalThreads_ * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    hostOutput_.allocate(outputSize_, cudaHostAllocMapped);

    initialized_ = true;
}

// Kontynuacja GPUEngine.cu
void Engine::cleanup() {
    if (computeStream_) cudaStreamDestroy(computeStream_);
    if (memcpyStream_) cudaStreamDestroy(memcpyStream_);
    if (computeDoneEvent_) cudaEventDestroy(computeDoneEvent_);
}

void Engine::SetPrefixes(const std::vector<uint16_t>& prefixes) {
    if (!initialized_) throw std::runtime_error("Engine not initialized");
    
    std::memset(hostPrefixes_.get(), 0, _64K * 2);
    auto hostPtr = static_cast<uint16_t*>(hostPrefixes_.get());
    for (auto prefix : prefixes) hostPtr[prefix] = 1;
    
    checkCudaError(cudaMemcpy(devicePrefixes_.get(), hostPrefixes_.get(), 
                  _64K * 2, cudaMemcpyHostToDevice), "Copy prefixes");
    
    if (!rekeyEnabled_) hostPrefixes_.free();
}

bool Engine::SetKeys(const std::vector<Point>& points) {
    if (!initialized_) return false;
    if (points.size() < static_cast<size_t>(totalThreads_)) {
        throw std::runtime_error("Insufficient points for threads");
    }

    auto hostPtr = static_cast<uint64_t*>(hostKeys_.get());
    for (int i = 0; i < totalThreads_; i++) {
        const auto& point = points[i];
        for (int j = 0; j < 4; j++) {
            hostPtr[i * 8 + j] = point.x[j];
            hostPtr[i * 8 + j + 4] = point.y[j];
        }
    }

    checkCudaError(cudaMemcpy(deviceKeys_.get(), hostKeys_.get(), 
                  totalThreads_ * 32 * 2, cudaMemcpyHostToDevice), "Copy keys");

    if (!rekeyEnabled_) hostKeys_.free();
    return callKernel();
}

bool Engine::callKernel() {
    if (!initialized_) return false;

    dim3 grid(totalThreads_ / threadsPerGroup_);
    dim3 block(threadsPerGroup_);

    void* kernelFunc = nullptr;
    if (searchType_ == SearchType::P2SH) {
        kernelFunc = hasPattern_ ? (void*)comp_keys_p2sh_pattern : (void*)comp_keys_p2sh;
    } else {
        kernelFunc = hasPattern_ ? (void*)comp_keys_pattern : 
                   (searchMode_ == SearchMode::Compressed) ? (void*)comp_keys_comp : (void*)comp_keys;
    }

    void* args[] = {
        &searchMode_, &devicePrefixes_.get(), &devicePrefixLookup_.get(), 
        &deviceKeys_.get(), &maxFound_, &deviceOutput_.get()
    };

    checkCudaError(cudaLaunchKernel(kernelFunc, grid, block, args, 0, computeStream_), "Launch kernel");
    return true;
}

bool Engine::Launch(std::vector<Item>& foundItems, bool spinWait) {
    if (!initialized_) return false;
    foundItems.clear();

    if (spinWait) {
        context_->synchronize();
        checkCudaError(cudaMemcpy(hostOutput_.get(), deviceOutput_.get(), 
                      outputSize_, cudaMemcpyDeviceToHost), "Copy output");
    } else {
        checkCudaError(cudaMemcpyAsync(hostOutput_.get(), deviceOutput_.get(), 
                      4, cudaMemcpyDeviceToHost, memcpyStream_), "Async copy");
        checkCudaError(cudaStreamSynchronize(memcpyStream_), "Sync stream");
    }

    auto hostPtr = static_cast<uint32_t*>(hostOutput_.get());
    uint32_t foundCount = hostPtr[0];
    if (foundCount > maxFound_) foundCount = maxFound_;

    if (foundCount > 0) {
        checkCudaError(cudaMemcpy(hostOutput_.get(), deviceOutput_.get(), 
                      foundCount * ITEM_SIZE + 4, cudaMemcpyDeviceToHost), "Copy results");

        foundItems.reserve(foundCount);
        for (uint32_t i = 0; i < foundCount; i++) {
            uint32_t* itemPtr = hostPtr + (i * ITEM_SIZE32 + 1);
            Item item;
            item.threadId = itemPtr[0];
            item.endomorphism = reinterpret_cast<int16_t*>(&itemPtr[1])[0] & 0x7FFF;
            item.mode = (reinterpret_cast<int16_t*>(&itemPtr[1])[0] & 0x8000) != 0;
            item.increment = reinterpret_cast<int16_t*>(&itemPtr[1])[1];
            item.hash = reinterpret_cast<uint8_t*>(itemPtr + 2);
            foundItems.push_back(item);
        }
    }
    return callKernel();
}

void Engine::PrintCudaInfo() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found\n");
        return;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("GPU #%d %s (%dx%d cores) (%.1f MB)\n",
               i, props.name, props.multiProcessorCount,
               convertSMVerToCores(props.major, props.minor),
               props.totalGlobalMem / 1048576.0);
    }
}
