#include "BenchmarkSuite.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#ifdef __linux__
#include <sys/sysinfo.h>
#elif _WIN32
#include <windows.h>
#endif

using namespace std::chrono;

class BenchmarkSuite::Impl {
public:
    Impl(PerformanceTracker& tracker) 
        : tracker(tracker),
          cpuThreads(std::thread::hardware_concurrency()),
          gpuEnabled(false) {
        DetectHardware();
    }

    void RunAllTests() {
        TestCPUPerformance();
        if(gpuEnabled) TestGPUPerformance();
        TestMemoryThroughput();
        TestDiskIO();
        if(networkEnabled) TestNetworkLatency();
        ExportResults();
    }

private:
    void DetectHardware() {
        // Wykrywanie CPU
        cpuName = GetCPUName();
        cpuCores = cpuThreads;

        // Wykrywanie GPU
        #ifdef USE_CUDA
        cudaDeviceProp prop;
        if(cudaGetDeviceCount(&gpuCount) == cudaSuccess && gpuCount > 0) {
            cudaGetDeviceProperties(&prop, 0);
            gpuName = prop.name;
            gpuEnabled = true;
        }
        #endif

        // Wykrywanie pamięci
        totalMemory = GetTotalMemory();

        // Wykrywanie dysku
        diskSize = GetDiskSpace();

        networkEnabled = CheckNetwork();
    }

    void TestCPUPerformance() {
        const int AES_ITERATIONS = 10'000'000;
        const int HASH_ITERATIONS = 1'000'000;

        auto aesTime = Measure([&] {
            volatile int result = 0;
            for(int i = 0; i < AES_ITERATIONS; ++i) {
                result ^= PerformAESRound(i);
            }
        });

        auto hashTime = Measure([&] {
            std::vector<uint8_t> data(1024);
            for(int i = 0; i < HASH_ITERATIONS; ++i) {
                PerformSHA256(data);
            }
        });

        tracker.AddResult("CPU AES", AES_ITERATIONS / aesTime.count(), "ops/ms");
        tracker.AddResult("CPU SHA256", HASH_ITERATIONS / hashTime.count(), "hashes/ms");
    }

    void TestGPUPerformance() {
        #ifdef USE_CUDA
        const int CUDA_ITERATIONS = 100'000'000;
        
        auto time = Measure([&] {
            GPUKernelBenchmark(CUDA_ITERATIONS);
        });

        tracker.AddResult("GPU Crypto", CUDA_ITERATIONS / time.count(), "ops/ms");
        #endif
    }

    void TestMemoryThroughput() {
        const size_t SIZE = 1GB;
        auto buffer = std::make_unique<uint8_t[]>(SIZE);

        auto writeTime = Measure([&] {
            std::memset(buffer.get(), 0xAA, SIZE);
        });

        auto readTime = Measure([&] {
            volatile auto sink = std::accumulate(buffer.get(), buffer.get() + SIZE, 0);
        });

        tracker.AddResult("Memory Write", SIZE / (writeTime.count() * 1MB), "GB/s");
        tracker.AddResult("Memory Read", SIZE / (readTime.count() * 1MB), "GB/s");
    }

    void TestDiskIO() {
        const size_t FILE_SIZE = 1GB;
        const std::string TEST_FILE = "benchmark.tmp";

        // Test zapisu
        auto writeTime = Measure([&] {
            std::ofstream file(TEST_FILE, std::ios::binary);
            std::vector<uint8_t> buffer(1MB);
            for(size_t i = 0; i < FILE_SIZE / 1MB; ++i) {
                file.write(reinterpret_cast<char*>(buffer.data()), buffer.size());
            }
        });

        // Test odczytu
        auto readTime = Measure([&] {
            std::ifstream file(TEST_FILE, std::ios::binary);
            std::vector<uint8_t> buffer(1MB);
            while(file.read(reinterpret_cast<char*>(buffer.data()), buffer.size()));
        });

        std::remove(TEST_FILE.c_str());

        tracker.AddResult("Disk Write", FILE_SIZE / (writeTime.count() * 1MB), "GB/s");
        tracker.AddResult("Disk Read", FILE_SIZE / (readTime.count() * 1MB), "GB/s");
    }

    template<typename F>
    auto Measure(F&& func) {
        auto start = high_resolution_clock::now();
        func();
        return high_resolution_clock::now() - start;
    }

    void ExportResults() {
        ExportJSON();
        ExportMarkdown();
    }

    void ExportJSON() {
        json report;
        report["hardware"] = {
            {"cpu", cpuName},
            {"cores", cpuCores},
            {"gpu", gpuName},
            {"memory", totalMemory},
            {"disk", diskSize}
        };
        
        auto results = tracker.GetResults();
        for(const auto& [name, metric] : results) {
            report["results"][name] = metric;
        }

        std::ofstream("benchmark.json") << report.dump(4);
    }

    void ExportMarkdown() {
        std::ofstream md("BENCHMARKS.md");
        md << "# Performance Benchmark Report\n\n";
        md << "## Hardware Configuration\n";
        md << "- CPU: " << cpuName << " (" << cpuCores << " threads)\n";
        md << "- GPU: " << (gpuEnabled ? gpuName : "None") << "\n";
        md << "- Memory: " << totalMemory / 1GB << " GB\n";
        md << "- Disk: " << diskSize / 1GB << " GB\n\n";

        md << "## Performance Metrics\n";
        md << "| Test | Result |\n|-----|-----|\n";
        for(const auto& [name, metric] : tracker.GetResults()) {
            md << "| " << name << " | " << metric.value << " " << metric.unit << " |\n";
        }
    }

    // Funkcje pomocnicze
    std::string GetCPUName() {
        #ifdef __linux__
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while(getline(cpuinfo, line)) {
            if(line.find("model name") != std::string::npos) {
                return line.substr(line.find(":") + 2);
            }
        }
        return "Unknown CPU";
        #elif _WIN32
        char buffer[128];
        DWORD size = sizeof(buffer);
        RegGetValue(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                   "ProcessorNameString", RRF_RT_REG_SZ, NULL, buffer, &size);
        return buffer;
        #endif
    }

    size_t GetTotalMemory() {
        #ifdef __linux__
        struct sysinfo info;
        sysinfo(&info);
        return info.totalram;
        #elif _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        return status.ullTotalPhys;
        #endif
    }

    // Członkowie klasy
    PerformanceTracker& tracker;
    std::string cpuName;
    std::string gpuName;
    int cpuCores;
    int cpuThreads;
    int gpuCount = 0;
    size_t totalMemory;
    size_t diskSize;
    bool gpuEnabled;
    bool networkEnabled;
};

// Interfejs publiczny
BenchmarkSuite::BenchmarkSuite(PerformanceTracker& tracker)
    : impl(new Impl(tracker)) {}

BenchmarkSuite::~BenchmarkSuite() = default;

void BenchmarkSuite::RunAllTests() { impl->RunAllTests(); }
