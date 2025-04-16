#include "VanitySearch.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <queue>

// Nowe stałe optymalizacyjne
constexpr uint64_t GPU_BATCH_SIZE = 1'000'000;
constexpr uint64_t CPU_BATCH_SIZE = 100'000;
constexpr int STATUS_UPDATE_MS = 250;

// Globalne liczniki i statystyki
static std::atomic<uint64_t> totalKeysChecked{0};
static std::atomic<bool> solutionFound{false};

// Współdzielona kolejka zadań z lock-free dostępem
class TaskQueue {
public:
    void Push(const KeyRange& range) {
        std::lock_guard<std::mutex> lock(mutex);
        ranges.push(range);
    }

    bool Pop(KeyRange& range) {
        std::lock_guard<std::mutex> lock(mutex);
        if(ranges.empty()) return false;
        range = ranges.front();
        ranges.pop();
        return true;
    }

private:
    std::queue<KeyRange> ranges;
    std::mutex mutex;
};

VanitySearch::VanitySearch(const Parameters& params) : 
    params(params),
    gpuEngine(params.deviceId),
    cpuEngine(params.threadCount) {
    
    // Inicjalizacja hybrydowego środowiska
    InitPatternMatching(params.patterns);
    InitKeySpace(params.startKey, params.endKey);
    
    if(params.useGPU) {
        gpuEngine.Init();
        gpuEngine.SetPatterns(params.patterns);
    }
}

void VanitySearch::Run() {
    // Inicjalizacja wątków
    std::vector<std::thread> workers;
    TaskQueue taskQueue;

    // Generowanie zadań
    GenerateInitialTasks(taskQueue);

    // Uruchomienie workerów
    for(int i = 0; i < params.threadCount; ++i) {
        workers.emplace_back([&]() {
            WorkerThread(taskQueue);
        });
    }

    // Monitoring postępu
    ProgressReporter reporter;
    reporter.Run();

    // Oczekiwanie na zakończenie
    for(auto& t : workers) t.join();
    reporter.Stop();
}

void VanitySearch::WorkerThread(TaskQueue& queue) {
    KeyRange range;
    while(!solutionFound && queue.Pop(range)) {
        if(params.useGPU) {
            ProcessGPUBatch(range);
        } else {
            ProcessCPUBatch(range);
        }
        totalKeysChecked += range.count;
    }
}

void VanitySearch::ProcessGPUBatch(const KeyRange& range) {
    std::vector<uint256_t> keys;
    keys.reserve(GPU_BATCH_SIZE);

    while(range.current < range.end) {
        // Generowanie batcha dla GPU
        uint64_t count = std::min(GPU_BATCH_SIZE, range.end - range.current);
        GenerateKeys(range.current, count, keys);
        
        // Przetwarzanie na GPU
        auto results = gpuEngine.Compute(keys);
        
        // Sprawdzanie wyników
        if(CheckResults(results)) {
            solutionFound = true;
            return;
        }
        
        range.current += count;
    }
}

void VanitySearch::ProcessCPUBatch(const KeyRange& range) {
    std::vector<uint256_t> keys(CPU_BATCH_SIZE);
    std::vector<uint32_t> results(CPU_BATCH_SIZE * 2);

    while(range.current < range.end) {
        uint64_t count = std::min(CPU_BATCH_SIZE, range.end - range.current);
        GenerateKeys(range.current, count, keys.data());
        
        cpuEngine.Compute(keys.data(), count, results.data());
        
        if(CheckResults(results.data(), count)) {
            solutionFound = true;
            return;
        }
        
        range.current += count;
    }
}

class ProgressReporter {
public:
    void Run() {
        reporterThread = std::thread([this]() {
            using namespace std::chrono;
            auto start = high_resolution_clock::now();
            
            while(!stop) {
                auto now = high_resolution_clock::now();
                auto elapsed = duration_cast<milliseconds>(now - start).count();
                
                if(elapsed >= STATUS_UPDATE_MS) {
                    PrintStatus(elapsed);
                    start = now;
                }
                
                std::this_thread::sleep_for(1ms);
            }
        });
    }

    void Stop() {
        stop = true;
        if(reporterThread.joinable()) reporterThread.join();
        PrintStatus(0);
    }

private:
    void PrintStatus(int64_t elapsedMs) {
        uint64_t keys = totalKeysChecked;
        double speed = keys / (elapsedMs / 1000.0);
        
        std::cout << "\r[STATUS] Keys: " << keys 
                  << " Speed: " << HumanReadableSpeed(speed) 
                  << " Elapsed: " << FormatTime(elapsedMs) << std::flush;
    }

    std::thread reporterThread;
    std::atomic<bool> stop{false};
};

// Nowe funkcje pomocnicze
std::string HumanReadableSpeed(double keysPerSec) {
    const char* units[] = {"Kkeys/s", "Mkeys/s", "Gkeys/s"};
    size_t i = 0;
    
    while(keysPerSec >= 1000 && i < 3) {
        keysPerSec /= 1000;
        i++;
    }
    
    return std::to_string(keysPerSec) + " " + units[i];
}

std::string FormatTime(int64_t ms) {
    auto sec = ms / 1000;
    return std::to_string(sec / 3600) + "h " 
         + std::to_string((sec % 3600) / 60) + "m " 
         + std::to_string(sec % 60) + "s";
}
