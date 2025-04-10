/*
 * Zmodernizowana wersja GPUEngine.h dla VanitySearch
 * Nowe optymalizacje: wsparcie CUDA 12+, Ampere/Ada Lovelace
 * Zachowano pełną kompatybilność wsteczną
 */

#ifndef GPUENGINE_H
#define GPUENGINE_H

#include <vector>
#include <string>
#include <cstdint>
#include "../SECP256k1.h"

// Nowe stałe dla architektur Ampere/Ada
#define SM80_OPTIMIZATIONS 1
#define MAX_RESULTS_DEFAULT 1000000

// Tryby wyszukiwania
enum SearchMode {
    SEARCH_COMPRESSED = 0,
    SEARCH_UNCOMPRESSED = 1,
    SEARCH_BOTH = 2
};

// Tryby przeszukiwania
enum SearchType {
    SEARCH_TYPE_PREFIX = 0,
    SEARCH_TYPE_SUFFIX = 1,
    SEARCH_TYPE_BOTH = 2
};

static const char *SEARCH_MODE_STR[] = {
    "Compressed",
    "Uncompressed",
    "Compressed or Uncompressed"
};

// Nowe parametry wydajnościowe
#define STEP_SIZE 2048  // Zwiększono dla lepszego wykorzystania SM
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024  // Dla Ampere/Ada

// Optymalizacja struktury danych
typedef struct __align__(16) {
    uint32_t thId;
    int16_t  incr;
    int16_t  endo;
    uint8_t  *hash;
    bool     mode;
    uint32_t padding;  // Wyrównanie do 16 bajtów
} GPUItem;

// Nowa struktura dla prefiksów z optymalizacją pamięci
typedef struct __align__(8) {
    uint16_t sPrefix;
    uint32_t lPrefixCount;
    uint32_t lPrefixOffset;  // Offset w tablicy lPrefixes
} GPULPrefix;

class GPUEngine {
public:
    // Konstruktor z nowymi parametrami optymalizacji
    GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, 
              uint32_t maxFound = MAX_RESULTS_DEFAULT, bool rekey = false);
    ~GPUEngine();

    // Nowe metody optymalizacyjne
    void SetComputeCapability(int major, int minor);
    void SetStreamCount(int count);  // Wsparcie dla wielu strumieni

    // Zmodernizowane metody
    void SetPrefix(const std::vector<uint16_t>& prefixes);
    void SetPrefix(const std::vector<GPULPrefix>& prefixes, uint32_t totalPrefix);
    bool SetKeys(const Point *points, size_t count);
    void SetSearchMode(SearchMode mode);
    void SetSearchType(SearchType type);
    void SetPattern(const std::string& pattern);
    
    // Optymalizacja: asynchroniczna wersja Launch
    bool LaunchAsync(std::vector<GPUItem>& results, cudaStream_t stream = 0);
    bool Launch(std::vector<GPUItem>& results, bool spinWait = false);

    // Nowe metody diagnostyczne
    float GetOccupancy() const;
    size_t GetMemoryUsage() const;

    static void PrintCudaInfo();
    static void GenerateCode(Secp256K1 *secp, int size);

private:
    // Nowe metody pomocnicze
    void AllocateMemory();
    void ConfigureKernel();
    bool CheckHash(uint8_t *h, std::vector<GPUItem>& found, int tid, int incr, int endo, int *ok);

    // Zmodernizowane pola
    int m_computeCapabilityMajor;
    int m_computeCapabilityMinor;
    int m_streamCount;
    bool m_useTensorCores;
    
    // Optymalizacja pamięci
    uint16_t *m_devicePrefixes;
    uint32_t *m_devicePrefixLookup;
    uint64_t *m_deviceKeys;
    uint32_t *m_deviceResults;

    // Dane konfiguracyjne
    SearchMode m_searchMode;
    SearchType m_searchType;
    uint32_t m_maxResults;
    bool m_rekeyEnabled;
    std::string m_pattern;
    bool m_hasPattern;

    // Nowe pola wydajnościowe
    float m_occupancy;
    size_t m_memoryUsage;
    cudaStream_t *m_streams;
};

#endif // GPUENGINE_H
