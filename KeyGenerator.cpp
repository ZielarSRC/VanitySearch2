#include "KeyGenerator.h"
#include <immintrin.h>
#include <openssl/sha.h>
#include <omp.h>

// Stałe optymalizacyjne
constexpr size_t KEY_BATCH_SIZE = 1'048'576;  // 1MB batch
constexpr int AES_ROUNDS = 10;
constexpr size_t SIMD_WIDTH = 16;

// Struktura zoptymalizowana pod SIMD
struct alignas(64) SIMDKeyBatch {
    __m512i keys[SIMD_WIDTH];
    uint64_t counters[SIMD_WIDTH];
};

class KeyGenerator::Impl {
public:
    Impl(const uint256_t& startKey, bool useHardware) : 
        currentKey(startKey),
        useHW(useHardware) {
        
        InitHardwareAcceleration();
        InitCounterMatrix();
    }

    void GenerateBatch(std::vector<uint256_t>& output) {
        if(useHW && hasAESNI) {
            GenerateSIMDBatch(output);
        } else {
            GenerateSerialBatch(output);
        }
    }

    void SetPatternMatcher(std::shared_ptr<PatternMatcher> matcher) {
        patternMatcher = matcher;
    }

    void SetRange(const uint256_t& end) {
        rangeEnd = end;
    }

private:
    void InitHardwareAcceleration() {
        // Wykrywanie funkcji CPU
        unsigned int eax, ebx, ecx, edx;
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        hasAESNI = (ecx & bit_AES);
        hasSHAEXT = (ecx & bit_SHA);
        hasAVX512 = (ebx & bit_AVX512F);
    }

    void InitCounterMatrix() {
        // Inicjalizacja wektorów liczników
        #pragma omp simd
        for(size_t i = 0; i < SIMD_WIDTH; ++i) {
            counterMatrix.counters[i] = i;
        }
    }

    __attribute__((target("aes,avx512f")))
    void GenerateSIMDBatch(std::vector<uint256_t>& output) {
        SIMDKeyBatch batch;
        size_t generated = 0;

        // Generowanie równoległe z użyciem AES-NI
        while(generated < KEY_BATCH_SIZE && currentKey < rangeEnd) {
            #pragma omp simd
            for(size_t i = 0; i < SIMD_WIDTH; ++i) {
                batch.keys[i] = _mm512_aesenc_epi128(
                    _mm512_load_epi64(currentKey.data()),
                    _mm512_set1_epi64(counterMatrix.counters[i])
                );
                counterMatrix.counters[i]++;
            }

            // Sprawdzanie wzorców w locie
            #pragma omp parallel for
            for(size_t i = 0; i < SIMD_WIDTH; ++i) {
                uint256_t key;
                _mm512_store_epi64(key.data(), batch.keys[i]);
                if(patternMatcher->Match(GenerateAddress(key))) {
                    #pragma omp critical
                    results.push_back(key);
                }
            }

            currentKey += SIMD_WIDTH;
            generated += SIMD_WIDTH;
        }
    }

    std::string GenerateAddress(const uint256_t& key) {
        // Implementacja generowania adresu z optymalizacjami
        uint8_t hash[32];
        SHA256_CTX ctx;
        
        // Wykorzystanie rozszerzeń SHA jeśli dostępne
        if(hasSHAEXT) {
            SHA256_Init_HW(&ctx);
            SHA256_Update_HW(&ctx, key.data(), 32);
            SHA256_Final_HW(hash, &ctx);
        } else {
            SHA256_Init(&ctx);
            SHA256_Update(&ctx, key.data(), 32);
            SHA256_Final(hash, &ctx);
        }

        return EncodeBase58Check(hash);
    }

    void GenerateSerialBatch(std::vector<uint256_t>& output) {
        // Rezerwacja pamięci z wyprzedzeniem
        output.reserve(KEY_BATCH_SIZE);
        
        // Generowanie sekwencyjne z optymalizacją pamięci podręcznej
        for(size_t i = 0; i < KEY_BATCH_SIZE && currentKey < rangeEnd; ++i) {
            output.push_back(currentKey++);
        }
    }

    // Członkowie klasy
    uint256_t currentKey;
    uint256_t rangeEnd;
    bool useHW;
    bool hasAESNI;
    bool hasSHAEXT;
    bool hasAVX512;
    SIMDKeyBatch counterMatrix;
    std::shared_ptr<PatternMatcher> patternMatcher;
    std::vector<uint256_t> results;
};

// Interfejs publiczny
KeyGenerator::KeyGenerator(const uint256_t& startKey, bool useHardware) :
    impl(new Impl(startKey, useHardware)) {}

KeyGenerator::~KeyGenerator() = default;

void KeyGenerator::GenerateBatch(std::vector<uint256_t>& output) { impl->GenerateBatch(output); }
void KeyGenerator::SetPatternMatcher(std::shared_ptr<PatternMatcher> matcher) { impl->SetPatternMatcher(matcher); }
void KeyGenerator::SetRange(const uint256_t& end) { impl->SetRange(end); }
