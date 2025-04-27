#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "../SECP256k1.h"

namespace GPUEngine {

    enum class SearchMode {
        Compressed,
        Uncompressed,
        Both
    };

    enum class SearchType {
        P2PKH,
        P2SH,
        BECH32
    };

    static constexpr const char* searchModeStrings[] = {
        "Compressed",
        "Uncompressed",
        "Compressed or Uncompressed"
    };

    static constexpr size_t STEP_SIZE = 1024;
    static constexpr size_t ITEM_SIZE = 28;
    static constexpr size_t ITEM_SIZE32 = ITEM_SIZE / 4;
    static constexpr size_t _64K = 65536;

    using PrefixT = uint16_t;
    using PrefixLT = uint32_t;

    struct Item {
        uint32_t threadId;
        int16_t increment;
        int16_t endomorphism;
        uint8_t* hash;
        bool mode;
    };

    struct LPrefix {
        PrefixT shortPrefix;
        std::vector<PrefixLT> longPrefixes;
    };

    class Engine {
    public:
        Engine(int threadGroups, int threadsPerGroup, int gpuId, uint32_t maxFound, bool rekey);
        ~Engine();

        // Deleted copy/move operations
        Engine(const Engine&) = delete;
        Engine& operator=(const Engine&) = delete;
        Engine(Engine&&) = delete;
        Engine& operator=(Engine&&) = delete;

        void SetPrefixes(const std::vector<PrefixT>& prefixes);
        void SetPrefixes(const std::vector<LPrefix>& prefixes, uint32_t totalPrefix);
        bool SetKeys(const std::vector<Point>& points);
        void SetSearchMode(SearchMode mode);
        void SetSearchType(SearchType type);
        void SetPattern(const std::string& pattern);
        bool Launch(std::vector<Item>& foundItems, bool spinWait = false);

        int GetThreadCount() const;
        static int GetGroupSize();

        bool Check(Secp256K1* secp);
        std::string GetDeviceName() const;

        static void PrintCudaInfo();
        static void GenerateCode(Secp256K1* secp, int size);

    private:
        bool CallKernel();
        bool CheckHash(const uint8_t* hash, std::vector<Item>& foundItems, 
                      int threadId, int increment, int endomorphism, int* okCount);

        int totalThreads_;
        int threadsPerGroup_;
        PrefixT* devicePrefixes_;
        PrefixT* hostPrefixes_;
        uint32_t* devicePrefixLookup_;
        uint32_t* hostPrefixLookup_;
        uint64_t* deviceKeys_;
        uint64_t* hostKeys_;
        uint32_t* deviceOutput_;
        uint32_t* hostOutput_;
        bool initialized_;
        SearchMode searchMode_;
        SearchType searchType_;
        bool isLittleEndian_;
        bool lostWarning_;
        bool rekeyEnabled_;
        uint32_t maxFound_;
        uint32_t outputSize_;
        std::string searchPattern_;
        bool hasPattern_;
    };

} // namespace GPUEngine
