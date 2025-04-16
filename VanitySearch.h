#ifndef VANITYSEARCH_H
#define VANITYSEARCH_H

#include "SECP256k1.h"
#include "GPUEngine.h"
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <regex>

class VanitySearch {
public:
    struct Result {
        SECP256k1::uint256_t privateKey;
        std::string address;
        uint64_t foundTime;
    };

    VanitySearch(const std::vector<std::string>& patterns, 
                bool useGPU = false,
                int deviceId = 0,
                int threads = 0);
    
    ~VanitySearch();

    void Start();
    void Stop();
    std::vector<Result> GetResults() const;
    std::string GetStatus() const;

private:
    void WorkerThread();
    void GPUWorker();
    bool MatchPattern(const std::string& address) const;
    std::string GenerateAddress(const SECP256k1::uint256_t& pubX, 
                               const SECP256k1::uint256_t& pubY) const;
    SECP256k1::uint256_t IncrementPrivateKey() const;

    // Konfiguracja
    std::vector<std::regex> patterns_;
    bool useGPU_;
    int deviceId_;
    int cpuThreads_;
    
    // Stan wyszukiwania
    mutable std::mutex mutex_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> keysChecked_{0};
    std::vector<Result> results_;
    
    // Zarządzanie wątkami
    std::vector<std::thread> workers_;
    std::unique_ptr<GPUEngine> gpuEngine_;
    
    // Generator kluczy
    SECP256k1::uint256_t currentKey_;
    SECP256k1::uint256_t startKey_;
    SECP256k1::uint256_t endKey_;
};

#endif
