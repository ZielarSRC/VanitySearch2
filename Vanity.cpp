#include "Vanity.h"
#include "GPUEngine.h"
#include "Base58.h"
#include <thread>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <mutex>
#include <queue>
#include <atomic>

// Constants matching original implementation
constexpr int COMPRESSED_PUBLIC_KEY_SIZE = 33;
constexpr int UNCOMPRESSED_PUBLIC_KEY_SIZE = 65;
constexpr int MAX_PREFIX_LENGTH = 100;
constexpr int REKEY_INTERVAL = 1000000;

// Thread-safe output management
class OutputManager {
    std::mutex mutex_;
    std::ostream* output_;
    std::unique_ptr<std::ofstream> fileOutput_;
public:
    OutputManager(const std::string& filePath) {
        if (!filePath.empty()) {
            fileOutput_ = std::make_unique<std::ofstream>(filePath, std::ios::app);
            output_ = fileOutput_.get();
        } else {
            output_ = &std::cout;
        }
    }

    void write(const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        *output_ << message << std::flush;
    }
};

Vanity::Vanity(Secp256K1* secp, 
              const std::vector<std::string>& prefixes,
              const std::string& seed,
              SearchMode searchMode,
              bool gpuEnable,
              bool stopWhenFound,
              const std::string& outputFile,
              bool useSSE,
              uint32_t maxFound,
              uint64_t rekey,
              bool caseSensitive,
              const Point& startPubKey,
              bool paranoiacSeed)
    : secp_(secp),
      prefixes_(prefixes),
      seed_(seed),
      searchMode_(searchMode),
      gpuEnable_(gpuEnable),
      stopWhenFound_(stopWhenFound),
      outputFile_(outputFile),
      useSSE_(useSSE),
      maxFound_(maxFound),
      rekey_(rekey),
      caseSensitive_(caseSensitive),
      startPubKey_(startPubKey),
      paranoiacSeed_(paranoiacSeed) {
    // Initialize thread-safe components
    outputManager_ = std::make_unique<OutputManager>(outputFile_);
}

void Vanity::Search(int nbCPUThread, const std::vector<int>& gpuId, const std::vector<int>& gridSize) {
    std::vector<std::thread> threads;
    std::atomic<bool> shouldStop(false);
    std::atomic<uint32_t> foundCount(0);

    // GPU initialization (if enabled)
    if (gpuEnable_) {
#ifdef WITHGPU
        threads.emplace_back([this, &gpuId, &gridSize, &shouldStop, &foundCount]() {
            try {
                GPUEngine gpu(gridSize[0], gridSize[1], gpuId[0], maxFound_, useSSE_);
                gpu.SetSearchMode(searchMode_);
                
                // Initialize GPU with starting key
                Int privKey = InitializePrivateKey();
                gpu.SetKeys(privKey);

                // Main GPU search loop
                while (!shouldStop && foundCount < maxFound_) {
                    gpu.Launch(gridSize[0], gridSize[1]);
                    
                    // Process results
                    std::vector<GPUEngine::Result> results;
                    gpu.GetResults(results);
                    
                    for (const auto& result : results) {
                        ProcessFoundKey(result.privKey, result.pubKey, result.compressed);
                        if (++foundCount >= maxFound_) break;
                    }

                    // Rekey logic
                    if (rekey_ > 0 && gpu.GetCount() > rekey_ * REKEY_INTERVAL) {
                        privKey.Add(gpu.GetCount());
                        gpu.SetKeys(privKey);
                    }
                }
            } catch (const std::exception& e) {
                outputManager_->write("GPU Error: " + std::string(e.what()) + "\n");
            }
        });
#else
        outputManager_->write("GPU support not available in this build\n");
#endif
    }

    // CPU threads initialization
    for (int i = 0; i < nbCPUThread; ++i) {
        threads.emplace_back([this, i, &shouldStop, &foundCount]() {
            try {
                Int privKey = InitializePrivateKey();
                if (i > 0) privKey.Add(i * 1000000); // Stagger threads

                // Main CPU search loop
                while (!shouldStop && foundCount < maxFound_) {
                    Point pubKey = secp_->ComputePublicKey(&privKey);
                    
                    // Check compressed address
                    if (searchMode_ != SEARCH_UNCOMPRESSED) {
                        std::string address = secp_->GetAddress(P2PKH, true, pubKey);
                        if (CheckAddress(address)) {
                            ProcessFoundKey(privKey, pubKey, true);
                            if (++foundCount >= maxFound_ || stopWhenFound_) break;
                        }
                    }
                    
                    // Check uncompressed address
                    if (searchMode_ != SEARCH_COMPRESSED) {
                        std::string address = secp_->GetAddress(P2PKH, false, pubKey);
                        if (CheckAddress(address)) {
                            ProcessFoundKey(privKey, pubKey, false);
                            if (++foundCount >= maxFound_ || stopWhenFound_) break;
                        }
                    }

                    privKey.Add(1);
                    
                    // Periodic check for stop condition
                    if (privKey.GetBitLength() % 64 == 0 && shouldStop) break;
                }
            } catch (const std::exception& e) {
                outputManager_->write("CPU Thread Error: " + std::string(e.what()) + "\n");
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        if (thread.joinable()) thread.join();
    }
}

Int Vanity::InitializePrivateKey() const {
    Int privKey;
    
    if (!seed_.empty()) {
        // Seed-based initialization (original PBKDF2 logic)
        std::string actualSeed = seed_;
        if (paranoiacSeed_) {
            actualSeed += Timer::getSeed(32);
        }
        
        unsigned char hseed[64];
        std::string salt = "VanitySearch";
        pbkdf2_hmac_sha512(hseed, sizeof(hseed),
            reinterpret_cast<const uint8_t*>(actualSeed.data()), actualSeed.size(),
            reinterpret_cast<const uint8_t*>(salt.data()), salt.size(),
            2048);
        
        privKey.SetInt32(0);
        sha256(hseed, sizeof(hseed), reinterpret_cast<uint8_t*>(privKey.bits64));
        
        if (!startPubKey_.isZero()) {
            // Adjust for custom start point
            privKey.Set(&secp_->order);
            privKey.Sub(&startPubKey_.x);
        }
    } else {
        // Default initialization
        privKey.SetBase16("0000000000000000000000000000000000000000000000000000000000000001");
    }
    
    return privKey;
}

bool Vanity::CheckAddress(const std::string& address) const {
    for (const auto& prefix : prefixes_) {
        if (Wildcard::match(address.c_str(), prefix.c_str(), caseSensitive_)) {
            return true;
        }
    }
    return false;
}

void Vanity::ProcessFoundKey(const Int& privKey, const Point& pubKey, bool compressed) {
    std::string address = secp_->GetAddress(P2PKH, compressed, pubKey);
    std::string privStr = secp_->GetPrivAddress(compressed, privKey);
    std::string pubStr = secp_->GetPublicKeyHex(compressed, pubKey);
    
    std::ostringstream oss;
    oss << "\nFound address: " << address << "\n"
        << "Private key (WIF): " << privStr << "\n"
        << "Private key (HEX): " << privKey.GetBase16() << "\n"
        << "Public key: " << pubStr << "\n\n";
    
    outputManager_->write(oss.str());
    
    if (stopWhenFound_) {
        std::lock_guard<std::mutex> lock(stopMutex_);
        shouldStop_ = true;
    }
}
