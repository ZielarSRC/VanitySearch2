// TurboSearchEngine.h
#pragma once
#include <x86intrin.h>
#include <array>
#include <vector>
#include <atomic>
#include <mutex>
#include <openssl/sha.h>
#include "SECP256k1.h"
#include "AVX256_SHA256.h"
#include "WorkStealingScheduler.h"

#ifdef WITH_GPU
#include "GPUEngine.h"
#include "GPUMemoryManager.h"
#endif

class TurboSearchEngine {
private:
    std::vector<std::thread> workers;
    std::atomic<uint64_t> keys_checked{0};
    std::atomic<bool> stop_flag{false};
    AVX256_SHA256 avx_hasher;
    SECP256k1 secp;
    WorkStealingScheduler scheduler;

#ifdef WITH_GPU
    GPUEngine gpu_engine;
    GPUMemoryManager gpu_memory;
#endif

    struct Result {
        Int key;
        std::array<uint8_t, 20> address;
    };

    std::mutex results_mutex;
    std::vector<Result> found_results;

    void cpu_search_worker(int thread_id) {
        constexpr size_t batch_size = 4;
        std::array<Int, batch_size> keys;
        std::array<Point, batch_size> points;
        std::array<std::array<uint8_t, 65>, batch_size> pubkeys;
        std::array<std::array<uint8_t, 32>, batch_size> hashes;

        SIMDKeyGenerator key_gen;
        MontgomeryLadder ladder;

        while (!stop_flag.load(std::memory_order_relaxed)) {
            // Generowanie kluczy
            key_gen.generate_4_keys(keys.data());

            // Obliczanie kluczy publicznych
            for (size_t i = 0; i < batch_size; ++i) {
                ladder.scalar_multiply(points[i], keys[i], secp.G());
                pubkeys[i][0] = 0x04;
                memcpy(pubkeys[i].data() + 1, points[i].x.bits64, 32);
                memcpy(pubkeys[i].data() + 33, points[i].y.bits64, 32);
            }

            // Haszowanie
            avx_hasher.hash_4_keys(pubkeys, hashes);

            // Sprawdzanie wyników
            for (size_t i = 0; i < batch_size; ++i) {
                if (check_address(hashes[i])) {
                    std::lock_guard<std::mutex> lock(results_mutex);
                    found_results.push_back({keys[i], hash_to_address(hashes[i])});
                }
            }

            keys_checked.fetch_add(batch_size, std::memory_order_relaxed);
        }
    }

#ifdef WITH_GPU
    void gpu_search_worker() {
        constexpr size_t batch_size = 1'000'000;
        std::vector<Int> keys(batch_size);
        std::vector<uint32_t> results(batch_size);

        while (!stop_flag.load(std::memory_order_relaxed)) {
            // Generowanie kluczy
            for (size_t i = 0; i < batch_size; ++i) {
                keys[i] = Int::Rand(256);
            }

            // Przeszukiwanie GPU
            gpu_engine.search(keys, results);

            // Przetwarzanie wyników
            for (size_t i = 0; i < batch_size; ++i) {
                if (results[i]) {
                    std::lock_guard<std::mutex> lock(results_mutex);
                    found_results.push_back({keys[i], {}});
                }
            }

            keys_checked.fetch_add(batch_size, std::memory_order_relaxed);
        }
    }
#endif

public:
    TurboSearchEngine() 
#ifdef WITH_GPU
        : gpu_memory(gpu_engine.get_context(), gpu_engine.get_device())
#endif
    {
        scheduler.initialize(std::thread::hardware_concurrency());
    }

    void start(int cpu_threads) {
#ifdef WITH_GPU
        if (gpu_engine.initialize()) {
            workers.emplace_back(&TurboSearchEngine::gpu_search_worker, this);
            cpu_threads = std::max(1, cpu_threads - 1);
        }
#endif

        for (int i = 0; i < cpu_threads; ++i) {
            workers.emplace_back(&TurboSearchEngine::cpu_search_worker, this, i);
        }
    }

    void stop() {
        stop_flag.store(true, std::memory_order_relaxed);
        for (auto& t : workers) {
            if (t.joinable()) t.join();
        }
    }

    std::vector<Result> get_results() const {
        std::lock_guard<std::mutex> lock(results_mutex);
        return found_results;
    }

    uint64_t get_keys_checked() const {
        return keys_checked.load(std::memory_order_relaxed);
    }
};
