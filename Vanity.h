/*
 * Modernized VanitySearch implementation
 * Optimized for contemporary CPUs and GPUs
 * Copyright (c) 2023 Modern Crypto Solutions
 */

#ifndef VANITY_SEARCH_H
#define VANITY_SEARCH_H

#include "SECP256k1.h"
#include "GPU/GPUEngine.h"
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <memory>

// Search modes
enum SearchMode {
    SEARCH_COMPRESSED,
    SEARCH_UNCOMPRESSED,
    SEARCH_BOTH
};

// Address types
enum AddressType {
    P2PKH = 0,
    P2SH = 1,
    BECH32 = 2
};

// Thread parameters
struct TH_PARAM {
    VanitySearch* obj;
    int threadId;
    std::atomic<bool> isRunning;
    std::atomic<bool> hasStarted;
    std::atomic<bool> rekeyRequest;
    int gridSizeX;
    int gridSizeY;
    int gpuId;
    Int THnextKey;
};

// Prefix item
struct PREFIX_ITEM {
    std::unique_ptr<char[]> prefix;
    int prefixLength;
    prefix_t sPrefix;
    double difficulty;
    std::unique_ptr<std::atomic<bool>> found;
    bool isFull;
    prefixl_t lPrefix;
    uint8_t hash160[20];
};

// Prefix table
struct PREFIX_TABLE_ITEM {
    std::unique_ptr<std::vector<PREFIX_ITEM>> items;
    std::atomic<bool> found;
};

// BitCrack parameters
struct BITCRACK_PARAM {
    Int ksStart;
    Int ksNext;
    Int ksFinish;
    int shareM;
    int shareN;
};

class VanitySearch {
public:
    VanitySearch(Secp256K1* secp, std::vector<std::string>& inputPrefixes, 
                std::string seed, int searchMode, bool useGpu, bool stop, 
                std::string outputFile, bool useSSE, uint32_t maxFound, 
                uint64_t rekey, bool caseSensitive, Point& startPubKey, 
                bool paranoiacSeed, std::string sessFile, BITCRACK_PARAM* bc);
    
    ~VanitySearch();
    
    void Search(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize);
    
private:
    // Internal methods
    void FindKeyCPU(TH_PARAM* p);
    void FindKeyGPU(TH_PARAM* p);
    
    std::string GetHex(std::vector<unsigned char>& buffer);
    std::string GetExpectedTime(double keyRate, double keyCount);
    std::string GetExpectedTimeBitCrack(double keyRate, double keyCount, BITCRACK_PARAM* bc);
    
    bool checkPrivKey(std::string addr, Int& key, int32_t incr, int endomorphism, bool mode);
    void checkAddr(int prefIdx, uint8_t* hash160, Int& key, int32_t incr, int endomorphism, bool mode);
    void checkAddrSSE(uint8_t* h1, uint8_t* h2, uint8_t* h3, uint8_t* h4, 
                     int32_t incr1, int32_t incr2, int32_t incr3, int32_t incr4,
                     Int& key, int endomorphism, bool mode);
    
    void checkAddresses(bool compressed, Int key, int i, Point p1);
    void checkAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4);
    
    void output(std::string addr, std::string pAddr, std::string pAddrHex);
    bool isAlive(TH_PARAM* p);
    bool hasStarted(TH_PARAM* p);
    void rekeyRequest(TH_PARAM* p);
    
    bool isSingularPrefix(std::string pref);
    bool initPrefix(std::string& prefix, PREFIX_ITEM* it);
    void dumpPrefixes();
    
    double getDiffuclty();
    void updateFound();
    
    void getCPUStartingKey(int thId, Int& key, Point& startP, uint64_t* tasksize, Int& THnextKey);
    void getGPUStartingKeys(int thId, int groupSize, int nbThread, Int* keys, Point* p, uint64_t* tasksize, Int& THnextKey);
    
    void enumCaseUnsentivePrefix(std::string s, std::vector<std::string>& list);
    bool prefixMatch(const char* prefix, const char* addr);
    void saveProgress(TH_PARAM* p, Int& lastSaveKey, BITCRACK_PARAM* bc);

    // Members
    Secp256K1* secp;
    Int startKey;
    Int IncrStartKey;
    Point startPubKey;
    bool startPubKeySpecified;
    
    std::atomic<uint64_t> counters[256];
    std::atomic<uint64_t> task_counters[256];
    
    double startTime;
    int searchType;
    int searchMode;
    bool hasPattern;
    bool caseSensitive;
    bool useGpu;
    bool stopWhenFound;
    std::atomic<bool> endOfSearch;
    
    int nbCPUThread;
    int nbGPUThread;
    std::atomic<int> nbFoundKey;
    
    uint64_t rekey;
    uint64_t lastRekey;
    uint32_t nbPrefix;
    
    std::string outputFile;
    bool useSSE;
    bool onlyFull;
    uint32_t maxFound;
    double _difficulty;
    
    std::unique_ptr<std::atomic<bool>[]> patternFound;
    std::vector<PREFIX_TABLE_ITEM> prefixes;
    std::vector<prefix_t> usedPrefix;
    std::vector<LPREFIX> usedPrefixL;
    std::vector<std::string>& inputPrefixes;
    
    std::string sessFile;
    BITCRACK_PARAM* bc;
    
    Int beta;
    Int lambda;
    Int beta2;
    Int lambda2;
    
    std::mutex outputMutex;
    std::mutex keyMutex;
};

#endif // VANITY_SEARCH_H
