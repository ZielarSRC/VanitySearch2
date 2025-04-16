#include "MemoryManager.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

class MemoryManager::Impl {
public:
    Impl(size_t bufferSize, int deviceId) : 
        deviceId(deviceId),
        bufferSize(bufferSize),
        cudaEnabled(true) {
        
        InitPools();
        InitUnifiedMemory();
    }

    void* Allocate(AllocType type, size_t size) {
        switch(type) {
            case CPU_PAGEABLE: return AllocHost(size);
            case CPU_PINNED: return AllocPinned(size);
            case GPU_DEVICE: return AllocDevice(size);
            case UNIFIED: return AllocUnified(size);
            default: throw std::runtime_error("Invalid allocation type");
        }
    }

    void Free(void* ptr, AllocType type) {
        std::lock_guard<std::mutex> lock(mtx);
        if(!ptr) return;

        switch(type) {
            case CPU_PAGEABLE: free(ptr); break;
            case CPU_PINNED: cudaFreeHost(ptr); break;
            case GPU_DEVICE: cudaFree(ptr); break;
            case UNIFIED: cudaFree(ptr); break;
        }

        allocationStats[type] -= GetSize(ptr);
    }

    void Copy(void* dst, const void* src, size_t size, CopyDirection dir) {
        cudaMemcpyKind kind;
        switch(dir) {
            case HOST_TO_HOST:     kind = cudaMemcpyHostToHost; break;
            case HOST_TO_DEVICE:   kind = cudaMemcpyHostToDevice; break;
            case DEVICE_TO_HOST:   kind = cudaMemcpyDeviceToHost; break;
            case DEVICE_TO_DEVICE: kind = cudaMemcpyDeviceToDevice; break;
        }

        cudaMemcpyAsync(dst, src, size, kind, currentStream);
    }

    void PrefetchToGPU(void* ptr, size_t size) {
        if(cudaEnabled)
            cudaMemPrefetchAsync(ptr, size, deviceId, currentStream);
    }

    void PrefetchToCPU(void* ptr, size_t size) {
        if(cudaEnabled)
            cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, currentStream);
    }

    void SetStream(cudaStream_t stream) {
        currentStream = stream;
    }

private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool inUse;
    };

    void InitPools() {
        // Inicjalizacja memory pooli dla każdego typu
        constexpr size_t INITIAL_POOL_SIZE = 1GB;
        
        InitPool(CPU_PINNED_POOL, INITIAL_POOL_SIZE);
        InitPool(GPU_DEVICE_POOL, INITIAL_POOL_SIZE);
        InitPool(UNIFIED_POOL, INITIAL_POOL_SIZE);
    }

    void InitPool(PoolType poolType, size_t size) {
        void* ptr;
        switch(poolType) {
            case CPU_PINNED_POOL:
                cudaMallocHost(&ptr, size);
                break;
            case GPU_DEVICE_POOL:
                cudaMalloc(&ptr, size);
                break;
            case UNIFIED_POOL:
                cudaMallocManaged(&ptr, size);
                break;
        }

        memoryPools[poolType].push_back({ptr, size, false});
    }

    void InitUnifiedMemory() {
        if(cudaEnabled) {
            cudaMemAdvise(unifiedMemory.data(), unifiedMemory.size(), 
                        cudaMemAdviseSetPreferredLocation, deviceId);
            cudaMemAdvise(unifiedMemory.data(), unifiedMemory.size(),
                        cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
        }
    }

    void* AllocHost(size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        auto& pool = memoryPools[CPU_PINNED_POOL];
        
        // Szukaj wolnego bloku w poolu
        for(auto& block : pool) {
            if(!block.inUse && block.size >= size) {
                block.inUse = true;
                return block.ptr;
            }
        }

        // Alokuj nowy blok jeśli brak
        void* ptr;
        cudaMallocHost(&ptr, size);
        pool.push_back({ptr, size, true});
        allocationStats[CPU_PINNED] += size;
        return ptr;
    }

    // Analogiczne implementacje dla innych typów alokacji

    std::mutex mtx;
    int deviceId;
    size_t bufferSize;
    bool cudaEnabled;
    cudaStream_t currentStream;
    
    enum PoolType {
        CPU_PINNED_POOL,
        GPU_DEVICE_POOL,
        UNIFIED_POOL
    };

    std::unordered_map<PoolType, std::vector<MemoryBlock>> memoryPools;
    std::unordered_map<AllocType, size_t> allocationStats;
    std::vector<uint8_t> unifiedMemory;
};

// Interfejs publiczny
MemoryManager::MemoryManager(size_t bufferSize, int deviceId) : 
    impl(new Impl(bufferSize, deviceId)) {}

MemoryManager::~MemoryManager() = default;

void* MemoryManager::Allocate(AllocType type, size_t size) { return impl->Allocate(type, size); }
void MemoryManager::Free(void* ptr, AllocType type) { impl->Free(ptr, type); }
void MemoryManager::Copy(void* dst, const void* src, size_t size, CopyDirection dir) { impl->Copy(dst, src, size, dir); }
void MemoryManager::PrefetchToGPU(void* ptr, size_t size) { impl->PrefetchToGPU(ptr, size); }
void MemoryManager::PrefetchToCPU(void* ptr, size_t size) { impl->PrefetchToCPU(ptr, size); }
void MemoryManager::SetStream(cudaStream_t stream) { impl->SetStream(stream); }
