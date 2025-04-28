// GPUEngine.h
#pragma once
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace GPUEngine {

    enum class SearchMode { Compressed, Uncompressed, Both };
    enum class SearchType { P2PKH, P2SH, BECH32 };

    struct Point { uint64_t x[4]; uint64_t y[4]; };
    struct Item { int threadId; int endomorphism; bool mode; int increment; uint8_t* hash; };

    class CudaContext {
    public:
        explicit CudaContext(int device_id);
        ~CudaContext();
        int device_id() const { return device_id_; }
        const cudaDeviceProp& properties() const { return prop_; }
        void synchronize() { cudaDeviceSynchronize(); }
    private:
        int device_id_;
        cudaDeviceProp prop_;
    };

    class GPUMemory {
    public:
        GPUMemory() = default;
        explicit GPUMemory(size_t size);
        ~GPUMemory();
        void* get() { return ptr_; }
        const void* get() const { return ptr_; }
        void allocate(size_t size);
        void free();
    private:
        void* ptr_ = nullptr;
        size_t size_ = 0;
    };

    class PinnedMemory {
    public:
        PinnedMemory() = default;
        explicit PinnedMemory(size_t size, unsigned int flags = cudaHostAllocDefault);
        ~PinnedMemory();
        void* get() { return ptr_; }
        const void* get() const { return ptr_; }
        void allocate(size_t size, unsigned int flags = cudaHostAllocDefault);
        void free();
    private:
        void* ptr_ = nullptr;
        size_t size_ = 0;
    };

    class Engine {
    public:
        Engine(int threadGroups, int threadsPerGroup, int gpuId, uint32_t maxFound, bool rekey);
        ~Engine();
        
        void SetPrefixes(const std::vector<uint16_t>& prefixes);
        void SetPattern(const std::string& pattern);
        bool SetKeys(const std::vector<Point>& points);
        bool Launch(std::vector<Item>& foundItems, bool spinWait);
        
        void SetSearchMode(SearchMode mode) { searchMode_ = mode; }
        void SetSearchType(SearchType type) { searchType_ = type; }
        
        bool Check(void* secp);
        static void PrintCudaInfo();
        std::string GetDeviceName() const { return deviceName_; }
        
    private:
        bool callKernel();
        void initialize();
        void cleanup();
        
        std::unique_ptr<CudaContext> context_;
        GPUMemory devicePrefixes_, devicePrefixLookup_, deviceKeys_, deviceOutput_;
        PinnedMemory hostPrefixes_, hostPrefixLookup_, hostKeys_, hostOutput_;
        
        cudaStream_t computeStream_ = 0;
        cudaStream_t memcpyStream_ = 0;
        cudaEvent_t computeDoneEvent_ = 0;
        
        int totalThreads_ = 0;
        int threadsPerGroup_ = 0;
        uint32_t maxFound_ = 0;
        size_t outputSize_ = 0;
        bool rekeyEnabled_ = false;
        bool initialized_ = false;
        bool hasPattern_ = false;
        SearchMode searchMode_;
        SearchType searchType_;
        std::string deviceName_;
        std::string searchPattern_;
    };
}
