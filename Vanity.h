#ifndef VANITY_H
#define VANITY_H

#include "SECP256k1.h"
#include "Wildcard.h"
#include "Timer.h"
#include <vector>
#include <string>
#include <string_view>
#include <memory>
#include <atomic>

class Vanity {
public:
    enum SearchMode {
        SEARCH_COMPRESSED = 0,
        SEARCH_UNCOMPRESSED = 1,
        SEARCH_BOTH = 2
    };

    Vanity(Secp256K1* secp, 
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
          bool paranoiacSeed);

    void Search(int nbCPUThread, const std::vector<int>& gpuId, const std::vector<int>& gridSize);

private:
    struct Config {
        std::vector<std::string> prefixes;
        std::string seed;
        SearchMode searchMode;
        bool gpuEnable;
        bool stopWhenFound;
        std::string outputFile;
        bool useSSE;
        uint32_t maxFound;
        uint64_t rekey;
        bool caseSensitive;
        Point startPubKey;
        bool paranoiacSeed;
    };

    Secp256K1* secp;
    Config config;
    std::atomic<bool> shouldStop{false};

    void SearchCPU();
    void SearchGPU(const std::vector<int>& gpuId, const std::vector<int>& gridSize);
    bool CheckAddress(const std::string& addr) const;
    void OutputResult(const Int& privKey, const Point& pubKey, const std::string& addr);
};

#endif // VANITY_H
