#include "GPUEngine.h"
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "../Timer.h"
#include "GPUGroup.h"
#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUBase58.h"
#include "GPUWildcard.h"
#include "GPUCompute.h"

namespace GPUEngine {

    namespace {
        constexpr const char* CUDA_ERROR_PREFIX = "CUDA error: ";

        void CheckCudaError(cudaError_t err, const char* context) {
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string(CUDA_ERROR_PREFIX) + 
                       context + ": " + cudaGetErrorString(err));
            }
        }

        int ConvertSMVerToCores(int major, int minor) {
            struct SMToCores {
                int version;
                int cores;
            };

            const SMToCores archCoresPerSM[] = {
                {0x20, 32}, {0x21, 48}, {0x30, 192}, {0x32, 192},
                {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128},
                {0x53, 128}, {0x60, 64}, {0x61, 128}, {0x62, 128},
                {0x70, 64}, {0x72, 64}, {0x75, 64}, {0x80, 64},
                {0x86, 128}, {-1, -1}
            };

            const int version = (major << 4) + minor;
            for (const auto& entry : archCoresPerSM) {
                if (entry.version == version) {
                    return entry.cores;
                }
            }
            return 0;
        }
    }

    Engine::Engine(int threadGroups, int threadsPerGroup, int gpuId, 
                  uint32_t maxFound, bool rekey) 
        : totalThreads_(0), threadsPerGroup_(threadsPerGroup),
          initialized_(false), rekeyEnabled_(rekey), maxFound_(maxFound),
          hasPattern_(false) {
        
        try {
            int deviceCount = 0;
            CheckCudaError(cudaGetDeviceCount(&deviceCount), "Get device count");

            if (deviceCount == 0) {
                throw std::runtime_error("No CUDA-capable devices available");
            }

            CheckCudaError(cudaSetDevice(gpuId), "Set device");

            cudaDeviceProp props;
            CheckCudaError(cudaGetDeviceProperties(&props, gpuId), "Get device properties");

            if (threadGroups == -1) {
                threadGroups = props.multiProcessorCount * 8;
            }

            totalThreads_ = threadGroups * threadsPerGroup;
            outputSize_ = (maxFound * ITEM_SIZE + 4);

            char buffer[512];
            snprintf(buffer, sizeof(buffer),
                    "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
                    gpuId, props.name, props.multiProcessorCount,
                    ConvertSMVerToCores(props.major, props.minor),
                    totalThreads_ / threadsPerGroup, threadsPerGroup);
            deviceName_ = buffer;

            // Configure device
            CheckCudaError(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1), "Set cache config");
            CheckCudaError(cudaDeviceSetLimit(cudaLimitStackSize, 49152), "Set stack size");

            // Allocate device memory
            CheckCudaError(cudaMalloc(&devicePrefixes_, _64K * 2), "Allocate prefixes");
            CheckCudaError(cudaHostAlloc(&hostPrefixes_, _64K * 2, 
                          cudaHostAllocWriteCombined | cudaHostAllocMapped), 
                          "Allocate pinned prefixes");
            
            CheckCudaError(cudaMalloc(&deviceKeys_, totalThreads_ * 32 * 2), "Allocate keys");
            CheckCudaError(cudaHostAlloc(&hostKeys_, totalThreads_ * 32 * 2,
                          cudaHostAllocWriteCombined | cudaHostAllocMapped),
                          "Allocate pinned keys");

            CheckCudaError(cudaMalloc(&deviceOutput_, outputSize_), "Allocate output");
            CheckCudaError(cudaHostAlloc(&hostOutput_, outputSize_, cudaHostAllocMapped),
                          "Allocate pinned output");

            searchMode_ = SearchMode::Compressed;
            searchType_ = SearchType::P2PKH;
            initialized_ = true;

        } catch (const std::exception& e) {
            Cleanup();
            throw;
        }
    }

    void Engine::Cleanup() {
        if (deviceKeys_) cudaFree(deviceKeys_);
        if (devicePrefixes_) cudaFree(devicePrefixes_);
        if (devicePrefixLookup_) cudaFree(devicePrefixLookup_);
        if (hostOutput_) cudaFreeHost(hostOutput_);
        if (deviceOutput_) cudaFree(deviceOutput_);
        if (hostPrefixes_) cudaFreeHost(hostPrefixes_);
        if (hostKeys_) cudaFreeHost(hostKeys_);
        if (hostPrefixLookup_) cudaFreeHost(hostPrefixLookup_);
    }

    Engine::~Engine() {
        Cleanup();
    }

        void Engine::SetPrefixes(const std::vector<PrefixT>& prefixes) {
        if (!initialized_) {
            throw std::runtime_error("Engine not initialized");
        }

        std::memset(hostPrefixes_, 0, _64K * 2);
        for (const auto prefix : prefixes) {
            hostPrefixes_[prefix] = 1;
        }

        CheckCudaError(cudaMemcpy(devicePrefixes_, hostPrefixes_, _64K * 2, cudaMemcpyHostToDevice),
                      "Copy prefixes to device");

        if (!rekeyEnabled_) {
            cudaFreeHost(hostPrefixes_);
            hostPrefixes_ = nullptr;
        }

        lostWarning_ = false;
    }

    void Engine::SetPrefixes(const std::vector<LPrefix>& prefixes, uint32_t totalPrefix) {
        if (!initialized_) {
            throw std::runtime_error("Engine not initialized");
        }

        // Allocate lookup memory
        CheckCudaError(cudaMalloc(&devicePrefixLookup_, (_64K + totalPrefix) * 4),
                      "Allocate prefix lookup");
        CheckCudaError(cudaHostAlloc(&hostPrefixLookup_, (_64K + totalPrefix) * 4,
                      cudaHostAllocWriteCombined | cudaHostAllocMapped),
                      "Allocate pinned prefix lookup");

        std::memset(hostPrefixes_, 0, _64K * 2);
        std::memset(hostPrefixLookup_, 0, _64K * 4);

        uint32_t offset = _64K;
        for (const auto& prefix : prefixes) {
            int count = static_cast<int>(prefix.longPrefixes.size());
            hostPrefixes_[prefix.shortPrefix] = static_cast<PrefixT>(count);
            hostPrefixLookup_[prefix.shortPrefix] = offset;
            
            for (const auto longPrefix : prefix.longPrefixes) {
                hostPrefixLookup_[offset++] = longPrefix;
            }
        }

        if (offset != (_64K + totalPrefix)) {
            throw std::runtime_error("Mismatch in total prefix count");
        }

        // Copy to device
        CheckCudaError(cudaMemcpy(devicePrefixes_, hostPrefixes_, _64K * 2, cudaMemcpyHostToDevice),
                      "Copy prefixes to device");
        CheckCudaError(cudaMemcpy(devicePrefixLookup_, hostPrefixLookup_, 
                      (_64K + totalPrefix) * 4, cudaMemcpyHostToDevice),
                      "Copy prefix lookup to device");

        // Free host memory
        cudaFreeHost(hostPrefixes_);
        hostPrefixes_ = nullptr;
        cudaFreeHost(hostPrefixLookup_);
        hostPrefixLookup_ = nullptr;
        
        lostWarning_ = false;
    }

    void Engine::SetPattern(const std::string& pattern) {
        if (!initialized_) {
            throw std::runtime_error("Engine not initialized");
        }

        if (pattern.size() >= _64K * 2) {
            throw std::runtime_error("Pattern too large");
        }

        std::memcpy(hostPrefixes_, pattern.data(), pattern.size());
        CheckCudaError(cudaMemcpy(devicePrefixes_, hostPrefixes_, _64K * 2, cudaMemcpyHostToDevice),
                      "Copy pattern to device");

        cudaFreeHost(hostPrefixes_);
        hostPrefixes_ = nullptr;
        
        lostWarning_ = false;
        hasPattern_ = true;
        searchPattern_ = pattern;
    }

    bool Engine::SetKeys(const std::vector<Point>& points) {
        if (!initialized_) {
            return false;
        }

        if (points.size() < static_cast<size_t>(totalThreads_)) {
            throw std::runtime_error("Insufficient points for threads");
        }

        for (int i = 0; i < totalThreads_; i += threadsPerGroup_) {
            for (int j = 0; j < threadsPerGroup_; j++) {
                const auto& point = points[i + j];
                const int baseIdx = 8 * i + j;

                hostKeys_[baseIdx + 0 * threadsPerGroup_] = point.x.bits64[0];
                hostKeys_[baseIdx + 1 * threadsPerGroup_] = point.x.bits64[1];
                hostKeys_[baseIdx + 2 * threadsPerGroup_] = point.x.bits64[2];
                hostKeys_[baseIdx + 3 * threadsPerGroup_] = point.x.bits64[3];
                hostKeys_[baseIdx + 4 * threadsPerGroup_] = point.y.bits64[0];
                hostKeys_[baseIdx + 5 * threadsPerGroup_] = point.y.bits64[1];
                hostKeys_[baseIdx + 6 * threadsPerGroup_] = point.y.bits64[2];
                hostKeys_[baseIdx + 7 * threadsPerGroup_] = point.y.bits64[3];
            }
        }

        CheckCudaError(cudaMemcpy(deviceKeys_, hostKeys_, totalThreads_ * 32 * 2, cudaMemcpyHostToDevice),
                      "Copy keys to device");

        if (!rekeyEnabled_) {
            cudaFreeHost(hostKeys_);
            hostKeys_ = nullptr;
        }

        return CallKernel();
    }

    bool Engine::CallKernel() {
        if (!initialized_) {
            return false;
        }

        // Reset found count
        CheckCudaError(cudaMemset(deviceOutput_, 0, 4), "Reset output buffer");

        const dim3 grid(totalThreads_ / threadsPerGroup_);
        const dim3 block(threadsPerGroup_);

        if (searchType_ == SearchType::P2SH) {
            if (hasPattern_) {
                comp_keys_p2sh_pattern<<<grid, block>>>(
                    static_cast<uint32_t>(searchMode_),
                    devicePrefixes_,
                    deviceKeys_,
                    maxFound_,
                    deviceOutput_);
            } else {
                comp_keys_p2sh<<<grid, block>>>(
                    static_cast<uint32_t>(searchMode_),
                    devicePrefixes_,
                    devicePrefixLookup_,
                    deviceKeys_,
                    maxFound_,
                    deviceOutput_);
            }
        } else {
            if (hasPattern_) {
                if (searchType_ == SearchType::BECH32) {
                    throw std::runtime_error("BECH32 not yet supported with wildcard");
                }
                comp_keys_pattern<<<grid, block>>>(
                    static_cast<uint32_t>(searchMode_),
                    devicePrefixes_,
                    deviceKeys_,
                    maxFound_,
                    deviceOutput_);
            } else {
                if (searchMode_ == SearchMode::Compressed) {
                    comp_keys_comp<<<grid, block>>>(
                        devicePrefixes_,
                        devicePrefixLookup_,
                        deviceKeys_,
                        maxFound_,
                        deviceOutput_);
                } else {
                    comp_keys<<<grid, block>>>(
                        static_cast<uint32_t>(searchMode_),
                        devicePrefixes_,
                        devicePrefixLookup_,
                        deviceKeys_,
                        maxFound_,
                        deviceOutput_);
                }
            }
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
        }

        return true;
    }

    bool Engine::Launch(std::vector<Item>& foundItems, bool spinWait) {
        if (!initialized_) {
            return false;
        }

        foundItems.clear();

        if (spinWait) {
            CheckCudaError(cudaMemcpy(hostOutput_, deviceOutput_, outputSize_, cudaMemcpyDeviceToHost),
                          "Copy output (spin wait)");
        } else {
            cudaEvent_t event;
            CheckCudaError(cudaEventCreate(&event), "Create event");
            CheckCudaError(cudaMemcpyAsync(hostOutput_, deviceOutput_, 4, cudaMemcpyDeviceToHost, 0),
                          "Async copy output");
            CheckCudaError(cudaEventRecord(event, 0), "Record event");

            while (cudaEventQuery(event) == cudaErrorNotReady) {
                Timer::SleepMillis(1);
            }
            CheckCudaError(cudaEventDestroy(event), "Destroy event");
        }

        uint32_t foundCount = hostOutput_[0];
        if (foundCount > maxFound_) {
            if (!lostWarning_) {
                printf("\nWarning, %d items lost\nHint: Search with less prefixes, less threads (-g) or increase maxFound (-m)\n",
                      (foundCount - maxFound_));
                lostWarning_ = true;
            }
            foundCount = maxFound_;
        }

        if (foundCount > 0) {
            CheckCudaError(cudaMemcpy(hostOutput_, deviceOutput_, foundCount * ITEM_SIZE + 4, cudaMemcpyDeviceToHost),
                          "Copy found items");

            foundItems.reserve(foundCount);
            for (uint32_t i = 0; i < foundCount; i++) {
                uint32_t* itemPtr = hostOutput_ + (i * ITEM_SIZE32 + 1);
                Item item;
                item.threadId = itemPtr[0];
                const int16_t* ptr = reinterpret_cast<int16_t*>(&itemPtr[1]);
                item.endomorphism = ptr[0] & 0x7FFF;
                item.mode = (ptr[0] & 0x8000) != 0;
                item.increment = ptr[1];
                item.hash = reinterpret_cast<uint8_t*>(itemPtr + 2);
                foundItems.push_back(item);
            }
        }

        return CallKernel();
    }

    bool Engine::CheckHash(const uint8_t* hash, std::vector<Item>& foundItems,
                         int threadId, int increment, int endomorphism, int* okCount) {
        auto it = std::find_if(foundItems.begin(), foundItems.end(),
            [hash](const Item& item) {
                return ripemd160_comp_hash(item.hash, hash);
            });

        if (it != foundItems.end()) {
            foundItems.erase(it);
            (*okCount)++;
            return true;
        }

        printf("Expected item not found %s (thread=%d, incr=%d, endo=%d)\n",
               toHex(hash, 20).c_str(), threadId, increment, endomorphism);
        return false;
    }

    bool Engine::Check(Secp256K1* secp) {
        if (!initialized_) {
            return false;
        }

        printf("GPU: %s\n", deviceName_.c_str());

#ifdef FULLCHECK
        // Verify endianness
        get_endianness<<<1, 1>>>(deviceOutput_);
        CheckCudaError(cudaGetLastError(), "Endianness check kernel");
        CheckCudaError(cudaMemcpy(hostOutput_, deviceOutput_, 1, cudaMemcpyDeviceToHost),
                      "Copy endianness result");
        isLittleEndian_ = *hostOutput_ != 0;
        printf("Endianness: %s\n", (isLittleEndian_ ? "Little" : "Big"));

        // Verify modular multiplication
        Int a, b, r, c;
        a.Rand(256);
        b.Rand(256);
        c.ModMulK1(&a, &b);
        
        std::memcpy(hostKeys_, a.bits64, BIFULLSIZE);
        std::memcpy(hostKeys_ + 5, b.bits64, BIFULLSIZE);
        CheckCudaError(cudaMemcpy(deviceKeys_, hostKeys_, BIFULLSIZE * 2, cudaMemcpyHostToDevice),
                      "Copy values for multiplication check");

        chekc_mult<<<1, 1>>>(deviceKeys_, deviceKeys_ + 5, (uint64_t*)deviceOutput_);
        CheckCudaError(cudaGetLastError(), "Multiplication check kernel");
        CheckCudaError(cudaMemcpy(hostOutput_, deviceOutput_, BIFULLSIZE, cudaMemcpyDeviceToHost),
                      "Copy multiplication result");
        std::memcpy(r.bits64, hostOutput_, BIFULLSIZE);

        if (!c.IsEqual(&r)) {
            printf("\nModular Mult wrong:\nR=%s\nC=%s\n",
                   toHex((uint8_t*)r.bits64, BIFULLSIZE).c_str(),
                   toHex((uint8_t*)c.bits64, BIFULLSIZE).c_str());
            return false;
        }

        // Verify hash computation
        uint8_t h[20], hc[20];
        Point pi;
        pi.x.Rand(256);
        pi.y.Rand(256);
        secp->GetHash160(pi, false, h);
        secp->GetHash160(pi, true, hc);

        std::memcpy(hostKeys_, pi.x.bits64, BIFULLSIZE);
        std::memcpy(hostKeys_ + 5, pi.y.bits64, BIFULLSIZE);
        CheckCudaError(cudaMemcpy(deviceKeys_, hostKeys_, BIFULLSIZE * 2, cudaMemcpyHostToDevice),
                      "Copy values for hash check");

        chekc_hash160<<<1, 1>>>(deviceKeys_, deviceKeys_ + 5, deviceOutput_);
        CheckCudaError(cudaGetLastError(), "Hash check kernel");
        CheckCudaError(cudaMemcpy(hostOutput_, deviceOutput_, 64, cudaMemcpyDeviceToHost),
                      "Copy hash results");

        if (!ripemd160_comp_hash((uint8_t*)hostOutput_, h)) {
            printf("\nGetHash160 wrong:\n%s\n%s\n",
                   toHex((uint8_t*)hostOutput_, 20).c_str(),
                   toHex(h, 20).c_str());
            return false;
        }

        if (!ripemd160_comp_hash((uint8_t*)(hostOutput_ + 5), hc)) {
            printf("\nGetHash160Comp wrong:\n%s\n%s\n",
                   toHex((uint8_t*)(hostOutput_ + 5), 20).c_str(),
                   toHex(h, 20).c_str());
            return false;
        }
#endif // FULLCHECK

        std::vector<Point> points(totalThreads_);
        std::vector<Point> points2(totalThreads_);
        Int k;

        if (searchMode_ == SearchMode::Both) {
            printf("Warning, Check function does not support BOTH_MODE, use either compressed or uncompressed");
            return true;
        }

        const bool searchComp = (searchMode_ == SearchMode::Compressed);
        const uint32_t seed = static_cast<uint32_t>(time(nullptr));
        printf("Seed: %u\n", seed);
        rseed(seed);

        int nbOK[6] = {0};
        int nbFoundCPU[6] = {0};
        std::vector<Item> foundItems;

        // Initialize points
        for (int i = 0; i < totalThreads_; i++) {
            k.Rand(256);
            points[i] = secp->ComputePublicKey(&k);
            k.Add((uint64_t)GRP_SIZE / 2);
            points2[i] = secp->ComputePublicKey(&k);
        }

        // Set test prefixes and keys
        SetPrefixes({0xFEFE, 0x1234});
        SetKeys(points2);
        
        const double t0 = Timer::get_tick();
        Launch(foundItems, true);
        const double t1 = Timer::get_tick();
        Timer::printResult("Key", 6 * STEP_SIZE * totalThreads_, t0, t1);

        printf("ComputeKeys() found %zu items, CPU check...\n", foundItems.size());

        // Prepare endomorphism constants
        Int beta, beta2;
        beta.SetBase16("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
        beta2.SetBase16("851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40");

        // Verify results
        bool ok = true;
        for (int j = 0; j < totalThreads_; j++) {
            for (int i = 0; i < STEP_SIZE; i++) {
                Point pt = points[j];
                Point p1 = points[j];
                Point p2 = points[j];
                p1.x.ModMulK1(&beta);
                p2.x.ModMulK1(&beta2);
                points[j] = secp->NextKey(points[j]);

                uint8_t h[20];
                auto checkPoint = [&](const Point& p, int endo, int incr, int idx) {
                    secp->GetHash160(SearchType::P2PKH, searchComp, p, h);
                    const PrefixT pr = *(PrefixT*)h;
                    if (pr == 0xFEFE || pr == 0x1234) {
                        nbFoundCPU[idx]++;
                        ok &= CheckHash(h, foundItems, j, incr, endo, nbOK + idx);
                    }
                };

                // Check all variations
                checkPoint(pt, 0, i, 0);
                checkPoint(p1, 1, i, 1);
                checkPoint(p2, 2, i, 2);

                // Check symmetric versions
                pt.y.ModNeg();
                p1.y.ModNeg();
                p2.y.ModNeg();

                checkPoint(pt, 0, -i, 3);
                checkPoint(p1, 1, -i, 4);
                checkPoint(p2, 2, -i, 5);
            }
        }

        if (ok && !foundItems.empty()) {
            ok = false;
            printf("Unexpected item found!\n");
        }

        if (!ok) {
            const int totalFound = std::accumulate(nbFoundCPU, nbFoundCPU + 6, 0);
            printf("CPU found %d items\n", totalFound);

            printf("GPU: point   correct [%d/%d]\n", nbOK[0], nbFoundCPU[0]);
            printf("GPU: endo #1 correct [%d/%d]\n", nbOK[1], nbFoundCPU[1]);
            printf("GPU: endo #2 correct [%d/%d]\n", nbOK[2], nbFoundCPU[2]);
            printf("GPU: sym/point   correct [%d/%d]\n", nbOK[3], nbFoundCPU[3]);
            printf("GPU: sym/endo #1 correct [%d/%d]\n", nbOK[4], nbFoundCPU[4]);
            printf("GPU: sym/endo #2 correct [%d/%d]\n", nbOK[5], nbFoundCPU[5]);
            printf("GPU/CPU check Failed!\n");
        } else {
            printf("GPU/CPU check OK\n");
        }

        return ok;
    }

    void Engine::PrintCudaInfo() {
        const char* computeModes[] = {
            "Multiple host threads",
            "Only one host thread",
            "No host thread",
            "Multiple process threads",
            "Unknown",
            nullptr
        };

        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess) {
            printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(err));
            return;
        }

        if (deviceCount == 0) {
            printf("GPUEngine: No available CUDA devices\n");
            return;
        }

        for (int i = 0; i < deviceCount; i++) {
            err = cudaSetDevice(i);
            if (err != cudaSuccess) {
                printf("GPUEngine: cudaSetDevice(%d) %s\n", i, cudaGetErrorString(err));
                continue;
            }

            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
            printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
                   i, props.name, props.multiProcessorCount,
                   ConvertSMVerToCores(props.major, props.minor),
                   props.major, props.minor,
                   props.totalGlobalMem / 1048576.0,
                   computeModes[props.computeMode]);
        }
    }

    int Engine::GetThreadCount() const {
        return totalThreads_;
    }

    int Engine::GetGroupSize() {
        return GRP_SIZE;
    }

    void Engine::SetSearchMode(SearchMode mode) {
        searchMode_ = mode;
    }

    void Engine::SetSearchType(SearchType type) {
        searchType_ = type;
    }

    std::string Engine::GetDeviceName() const {
        return deviceName_;
    }

    void Engine::GenerateCode(Secp256K1* secp, int size) {
        // Implementation depends on specific requirements
        throw std::runtime_error("GenerateCode not implemented in this version");
    }
} // namespace GPUEngine
