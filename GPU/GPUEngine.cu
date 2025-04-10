/*
 * Modernized GPUEngine for VanitySearch2
 * CUDA 12+ Optimized Version
 */

#include "GPUEngine.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
#include <cuda_arithmetic.h>
#include <vector>
#include <memory>

namespace cg = cooperative_groups;

// Modern C++ memory management
using DevicePtr = std::unique_ptr<void, decltype(&cudaFree)>;

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound, bool rekey) :
    rekey(rekey), nbThreadPerGroup(nbThreadPerGroup), initialised(false) {

    // Initialize CUDA with modern error handling
    cudaError_t err;
    int deviceCount = 0;
    
    auto cudaCheck = [](cudaError_t err) {
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    };

    try {
        // Device setup
        cudaCheck(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA capable devices found");
        }

        cudaCheck(cudaSetDevice(gpuId));
        
        cudaDeviceProp deviceProp;
        cudaCheck(cudaGetDeviceProperties(&deviceProp, gpuId));

        // Automatic configuration if needed
        if (nbThreadGroup == -1) {
            nbThreadGroup = deviceProp.multiProcessorCount * 8;
        }

        this->nbThread = nbThreadGroup * nbThreadPerGroup;
        this->maxFound = maxFound;
        this->outputSize = (maxFound * ITEM_SIZE + 4);

        // Modern string formatting
        deviceName = fmt::format("GPU #{} {} ({}x{} cores) Grid({}x{})",
            gpuId, deviceProp.name,
            deviceProp.multiProcessorCount,
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
            nbThread / nbThreadPerGroup,
            nbThreadPerGroup);

        // Configuration
        cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
        cudaCheck(cudaDeviceSetLimit(cudaLimitStackSize, 49152));

        // Modern memory allocation with RAII
        DevicePtr prefixPtr(nullptr, &cudaFree);
        DevicePtr prefixLookupPtr(nullptr, &cudaFree);
        DevicePtr keyPtr(nullptr, &cudaFree);
        DevicePtr outputPtr(nullptr, &cudaFree);

        void* tempPtr = nullptr;
        cudaCheck(cudaMalloc(&tempPtr, _64K * 2));
        prefixPtr.reset(tempPtr);

        cudaCheck(cudaMallocHost(&tempPtr, _64K * 2));
        inputPrefixPinned = static_cast<prefix_t*>(tempPtr);

        cudaCheck(cudaMalloc(&tempPtr, nbThread * 32 * 2));
        keyPtr.reset(tempPtr);

        cudaCheck(cudaMallocHost(&tempPtr, nbThread * 32 * 2));
        inputKeyPinned = static_cast<uint64_t*>(tempPtr);

        cudaCheck(cudaMalloc(&tempPtr, outputSize));
        outputPtr.reset(tempPtr);

        cudaCheck(cudaMallocHost(&tempPtr, outputSize));
        outputPrefixPinned = static_cast<uint32_t*>(tempPtr);

        // Transfer ownership to class members
        inputPrefix = static_cast<prefix_t*>(prefixPtr.release());
        inputKey = static_cast<uint64_t*>(keyPtr.release());
        outputPrefix = static_cast<uint32_t*>(outputPtr.release());

        searchMode = SEARCH_COMPRESSED;
        searchType = P2PKH;
        initialised = true;
        pattern = "";
        hasPattern = false;
        inputPrefixLookUp = nullptr;

    } catch (const std::exception& e) {
        std::cerr << "GPUEngine initialization failed: " << e.what() << std::endl;
        initialised = false;
    }
}

// Modernized memory management
GPUEngine::~GPUEngine() {
    if (initialised) {
        cudaFree(inputKey);
        cudaFree(inputPrefix);
        if (inputPrefixLookUp) cudaFree(inputPrefixLookUp);
        cudaFreeHost(outputPrefixPinned);
        cudaFree(outputPrefix);
    }
}

// Modern prefix setting with move semantics
void GPUEngine::SetPrefix(std::vector<prefix_t> prefixes) {
    std::memset(inputPrefixPinned, 0, _64K * 2);
    for (const auto& p : prefixes) {
        inputPrefixPinned[p] = 1;
    }

    cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);
    cudaFreeHost(inputPrefixPinned);
    inputPrefixPinned = nullptr;
    lostWarning = false;
}

void GPUEngine::SetPrefix(std::vector<LPREFIX> prefixes, uint32_t totalPrefix) {
    // Modern memory management
    DevicePtr lookupPtr(nullptr, &cudaFree);
    void* tempPtr = nullptr;
    
    cudaMalloc(&tempPtr, (_64K + totalPrefix) * 4);
    lookupPtr.reset(tempPtr);
    inputPrefixLookUp = static_cast<uint32_t*>(lookupPtr.release());

    cudaMallocHost(&tempPtr, (_64K + totalPrefix) * 4);
    inputPrefixLookUpPinned = static_cast<uint32_t*>(tempPtr);

    // Initialize data
    uint32_t offset = _64K;
    std::memset(inputPrefixPinned, 0, _64K * 2);
    std::memset(inputPrefixLookUpPinned, 0, _64K * 4);

    for (const auto& p : prefixes) {
        int nbLPrefix = static_cast<int>(p.lPrefixes.size());
        inputPrefixPinned[p.sPrefix] = static_cast<uint16_t>(nbLPrefix);
        inputPrefixLookUpPinned[p.sPrefix] = offset;
        
        std::copy(p.lPrefixes.begin(), p.lPrefixes.end(), 
                 inputPrefixLookUpPinned + offset);
        offset += nbLPrefix;
    }

    // Async memory transfers
    cudaMemcpyAsync(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(inputPrefixLookUp, inputPrefixLookUpPinned, 
                   (_64K + totalPrefix) * 4, cudaMemcpyHostToDevice);
    
    cudaFreeHost(inputPrefixPinned);
    inputPrefixPinned = nullptr;
    cudaFreeHost(inputPrefixLookUpPinned);
    inputPrefixLookUpPinned = nullptr;
    lostWarning = false;
}

// Modern kernel launch with error checking
bool GPUEngine::callKernel() {
    cudaMemset(outputPrefix, 0, 4);
    
    dim3 grid(nbThread / nbThreadPerGroup);
    dim3 block(nbThreadPerGroup);

    try {
        if (searchType == P2SH) {
            if (hasPattern) {
                comp_keys_p2sh_pattern<<<grid, block>>>(searchMode, inputPrefix, 
                                                       inputKey, maxFound, outputPrefix);
            } else {
                comp_keys_p2sh<<<grid, block>>>(searchMode, inputPrefix, 
                                              inputPrefixLookUp, inputKey, 
                                              maxFound, outputPrefix);
            }
        } else {
            if (hasPattern) {
                if (searchType == BECH32) {
                    throw std::runtime_error("BECH32 not yet supported with wildcard");
                }
                comp_keys_pattern<<<grid, block>>>(searchMode, inputPrefix, 
                                                 inputKey, maxFound, outputPrefix);
            } else {
                comp_keys<<<grid, block>>>(searchMode, inputPrefix, 
                                         inputPrefixLookUp, inputKey, 
                                         maxFound, outputPrefix);
            }
        }
        
        cudaDeviceSynchronize();
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Kernel launch failed: " << e.what() << std::endl;
        return false;
    }
}

// Modernized key setting with batch transfer
bool GPUEngine::SetKeys(Point *p) {
    const int elementsPerThread = 8;
    const size_t totalElements = nbThread * elementsPerThread;
    
    // Parallel copy using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < nbThread; i += nbThreadPerGroup) {
        for (int j = 0; j < nbThreadPerGroup; j++) {
            int idx = 8 * i + j;
            for (int k = 0; k < 4; k++) {
                inputKeyPinned[idx + k * nbThreadPerGroup] = p[i + j].x.bits64[k];
                inputKeyPinned[idx + (k + 4) * nbThreadPerGroup] = p[i + j].y.bits64[k];
            }
        }
    }

    // Async memory transfer
    cudaMemcpyAsync(inputKey, inputKeyPinned, nbThread * 32 * 2, 
                   cudaMemcpyHostToDevice);
    
    if (!rekey) {
        cudaFreeHost(inputKeyPinned);
        inputKeyPinned = nullptr;
    }

    return callKernel();
}

// Modern result handling with move semantics
bool GPUEngine::Launch(std::vector<ITEM> &prefixFound, bool spinWait) {
    prefixFound.clear();
    
    if (spinWait) {
        cudaMemcpy(outputPrefixPinned, outputPrefix, outputSize, 
                  cudaMemcpyDeviceToHost);
    } else {
        cudaEvent_t evt;
        cudaEventCreate(&evt);
        cudaMemcpyAsync(outputPrefixPinned, outputPrefix, 4, 
                       cudaMemcpyDeviceToHost);
        cudaEventRecord(evt);
        
        while (cudaEventQuery(evt) == cudaErrorNotReady) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        cudaEventDestroy(evt);
    }

    uint32_t nbFound = outputPrefixPinned[0];
    if (nbFound > maxFound) {
        if (!lostWarning) {
            fmt::print("\nWarning, {} items lost\nHint: Search with less prefixes, less threads (-g) or increase maxFound (-m)\n", 
                      (nbFound - maxFound));
            lostWarning = true;
        }
        nbFound = maxFound;
    }
    
    // Batch copy of results
    cudaMemcpy(outputPrefixPinned, outputPrefix, 
              nbFound * ITEM_SIZE + 4, cudaMemcpyDeviceToHost);

    // Reserve space for efficiency
    prefixFound.reserve(nbFound);
    
    for (uint32_t i = 0; i < nbFound; i++) {
        uint32_t *itemPtr = outputPrefixPinned + (i * ITEM_SIZE32 + 1);
        prefixFound.emplace_back(ITEM{
            .thId = itemPtr[0],
            .incr = static_cast<int16_t>(itemPtr[1] >> 16),
            .endo = static_cast<int16_t>(itemPtr[1] & 0x7FFF),
            .hash = reinterpret_cast<uint8_t*>(itemPtr + 2),
            .mode = (itemPtr[1] & 0x8000) != 0
        });
    }

    return callKernel();
}
