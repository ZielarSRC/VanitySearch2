#include "VanitySearch.h"
#include "SECP256k1.h"
#include "GPUEngine.h"
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <atomic>
#include <mutex>
#include <regex>
#include <iomanip>
#include <sstream>
#include <chrono>

using namespace std::chrono;

//================================================
// Stałe i zmienne globalne
//================================================
constexpr uint64_t GPU_BATCH_SIZE = 1'000'000;
constexpr uint64_t CPU_BATCH_SIZE = 100'000;
constexpr int STATUS_UPDATE_MS = 250;
constexpr char BASE58_CHARS[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

atomic<uint64_t> totalKeysChecked(0);
atomic<bool> solutionFound(false);

//================================================
// Implementacja klasy VanitySearch
//================================================

VanitySearch::VanitySearch(const Parameters& params) 
    : params(params),
      gpuEngine(params.deviceId),
      currentKey(params.startKey),
      startKey(params.startKey),
      endKey(params.endKey) {
    
    // Inicjalizacja wzorców
    for(const auto& pattern : params.patterns) {
        regexPatterns.emplace_back(
            ".*" + pattern + ".*", 
            regex_constants::optimize | regex_constants::icase
        );
    }
    
    if(params.useGPU) {
        gpuEngine.Init();
    }
}

void VanitySearch::Run() {
    vector<thread> workers;
    mutex queueMutex;
    queue<KeyRange> taskQueue;
    
    // Generowanie zadań
    KeyRange initialRange{startKey, endKey, startKey};
    taskQueue.push(initialRange);

    // Uruchomienie workerów
    for(int i = 0; i < params.threads; ++i) {
        workers.emplace_back([&]() {
            while(!solutionFound) {
                KeyRange range;
                {
                    lock_guard<mutex> lock(queueMutex);
                    if(taskQueue.empty()) break;
                    range = taskQueue.front();
                    taskQueue.pop();
                }
                
                if(params.useGPU) {
                    ProcessGPUBatch(range);
                } else {
                    ProcessCPUBatch(range);
                }
                
                totalKeysChecked += range.count;
            }
        });
    }

    // Monitorowanie postępu
    thread reporter([&]() {
        auto startTime = high_resolution_clock::now();
        while(!solutionFound) {
            auto now = high_resolution_clock::now();
            auto elapsed = duration_cast<milliseconds>(now - startTime).count();
            
            if(elapsed >= STATUS_UPDATE_MS) {
                PrintStatus(elapsed);
                startTime = now;
            }
            this_thread::sleep_for(1ms);
        }
    });

    // Czekaj na zakończenie
    for(auto& t : workers) t.join();
    solutionFound = true;
    reporter.join();
}

//================================================
// Metody prywatne
//================================================

void VanitySearch::ProcessGPUBatch(KeyRange& range) {
    vector<SECP256k1::uint256_t> keys;
    keys.reserve(GPU_BATCH_SIZE);

    while(range.current < range.end) {
        uint64_t count = min(GPU_BATCH_SIZE, range.end - range.current);
        GenerateKeys(range.current, count, keys);
        
        gpuEngine.SetKeys(keys);
        gpuEngine.Compute();
        auto results = gpuEngine.GetResults();
        
        if(CheckResults(results)) {
            solutionFound = true;
            return;
        }
        range.current += count;
    }
}

void VanitySearch::ProcessCPUBatch(KeyRange& range) {
    vector<SECP256k1::uint256_t> keys(CPU_BATCH_SIZE);
    vector<SECP256k1::Point> results(CPU_BATCH_SIZE);
    
    while(range.current < range.end) {
        uint64_t count = min(CPU_BATCH_SIZE, range.end - range.current);
        GenerateKeys(range.current, count, keys.data());
        
        #pragma omp parallel for
        for(uint64_t i = 0; i < count; ++i) {
            SECP256k1::Multiply(keys[i], results[i].x, results[i].y);
        }
        
        if(CheckResults(results)) {
            solutionFound = true;
            return;
        }
        range.current += count;
    }
}

void VanitySearch::GenerateKeys(const SECP256k1::uint256_t& start, uint64_t count, vector<SECP256k1::uint256_t>& keys) {
    keys.resize(count);
    SECP256k1::uint256_t current = start;
    
    for(uint64_t i = 0; i < count; ++i) {
        keys[i] = current;
        SECP256k1::Increment(current);
    }
}

bool VanitySearch::CheckResults(const vector<SECP256k1::Point>& results) {
    for(const auto& point : results) {
        string address = GenerateAddress(point.x, point.y);
        
        for(const auto& pattern : regexPatterns) {
            if(regex_search(address, pattern)) {
                lock_guard<mutex> lock(resultsMutex);
                results.push_back({
                    /* privateKey */ SECP256k1::uint256_t(), // Wypełnij odpowiednim kluczem
                    address,
                    duration_cast<seconds>(system_clock::now().time_since_epoch()).count()
                });
                return true;
            }
        }
    }
    return false;
}

string VanitySearch::GenerateAddress(const SECP256k1::uint256_t& x, const SECP256k1::uint256_t& y) const {
    // 1. Serializacja klucza publicznego
    vector<uint8_t> publicKey(65);
    publicKey[0] = 0x04;
    
    for(int i = 0; i < 8; ++i) {
        *reinterpret_cast<uint32_t*>(&publicKey[1 + i*4]) = htobe32(x.data[i]);
        *reinterpret_cast<uint32_t*>(&publicKey[33 + i*4]) = htobe32(y.data[i]);
    }

    // 2. SHA-256 + RIPEMD-160
    uint8_t sha256[SHA256_DIGEST_LENGTH];
    SHA256(publicKey.data(), publicKey.size(), sha256);
    
    uint8_t ripemd160[RIPEMD160_DIGEST_LENGTH];
    RIPEMD160(sha256, SHA256_DIGEST_LENGTH, ripemd160);

    // 3. Dodaj prefix sieciowy
    vector<uint8_t> payload;
    payload.reserve(21);
    payload.push_back(0x00); // Mainnet
    payload.insert(payload.end(), ripemd160, ripemd160 + 20);

    // 4. Oblicz checksum
    uint8_t checksum[SHA256_DIGEST_LENGTH];
    SHA256(payload.data(), payload.size(), checksum);
    SHA256(checksum, SHA256_DIGEST_LENGTH, checksum);

    // 5. Zbuduj pełny payload
    vector<uint8_t> fullPayload(payload);
    fullPayload.insert(fullPayload.end(), checksum, checksum + 4);

    // 6. Konwersja Base58
    string address;
    vector<int> digits;
    int leadingZeros = 0;
    
    // Konwersja z Big Endian
    while(fullPayload[leadingZeros] == 0) ++leadingZeros;
    
    CBigNum num;
    num.setvch(fullPayload);
    
    while(num > 0) {
        digits.push_back(num % 58);
        num /= 58;
    }
    
    address.append(leadingZeros, '1');
    for(auto it = digits.rbegin(); it != digits.rend(); ++it) {
        address += BASE58_CHARS[*it];
    }
    
    return address;
}

void VanitySearch::PrintStatus(int64_t elapsedMs) const {
    double speed = totalKeysChecked.load() / (elapsedMs / 1000.0);
    stringstream ss;
    
    ss << "\r[STATUS] Keys: " << totalKeysChecked.load()
       << " Speed: " << (speed > 1e6 ? speed/1e6 : speed/1e3)
       << (speed > 1e6 ? " Mkeys/s" : " Kkeys/s")
       << " Elapsed: " << format("{:%H:%M:%S}", floor<seconds>(system_clock::now() - startTime));
    
    cout << ss.str() << flush;
}

//================================================
// Funkcje pomocnicze
//================================================

string VanitySearch::HumanReadableSpeed(double keysPerSec) {
    const char* units[] = {"Kkeys/s", "Mkeys/s", "Gkeys/s"};
    int unitIndex = 0;
    
    while(keysPerSec >= 1000 && unitIndex < 2) {
        keysPerSec /= 1000;
        unitIndex++;
    }
    
    stringstream ss;
    ss << fixed << setprecision(2) << keysPerSec << " " << units[unitIndex];
    return ss.str();
}
