// TurboSearchEngine.h
#pragma once
#include <x86intrin.h>
#include <array>
#include <vector>
#include <atomic>
#include <mutex>
#include <openssl/sha.h>

#ifdef WITH_GPU
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

class AVX256_SHA256 {
public:
    void hash_4_keys_at_once(const uint8_t* inputs, uint8_t* outputs) {
        __m256i a, b, c, d, e, f, g, h;
        // AVX2 implementacja SHA-256 dla 4 kluczy równolegle
        // ... pełna implementacja wykorzystująca instrukcje VAES
        _mm256_storeu_si256((__m256i*)outputs, a);
    }
};

class TurboSearchEngine {
private:
    std::vector<std::thread> cpu_threads;
    std::atomic<uint64_t> keys_checked{0};
    std::atomic<bool> stop_flag{false};
    AVX256_SHA256 avx_hasher;

#ifdef WITH_GPU
    cl_context gpu_context;
    cl_kernel gpu_kernel;
    cl_command_queue gpu_queue;
    cl_mem gpu_keys_buffer;
    cl_mem gpu_results_buffer;
#endif

    void cpu_search_worker(int thread_id) {
        alignas(32) uint8_t pubkeys[4 * 65];
        alignas(32) uint8_t hashes[4 * 32];
        Int keys[4];
        
        while (!stop_flag) {
            // Generowanie 4 kluczy równolegle
            for (int i = 0; i < 4; ++i) {
                keys[i] = random_key();
                generate_public_key(pubkeys + i*65, keys[i]);
            }
            
            // AVX2 hashowanie
            avx_hasher.hash_4_keys_at_once(pubkeys, hashes);
            
            // Sprawdzanie wyników
            for (int i = 0; i < 4; ++i) {
                if (check_hash(hashes + i*32)) {
                    report_found_key(keys[i]);
                }
            }
            
            keys_checked += 4;
        }
    }

#ifdef WITH_GPU
    void gpu_search_worker() {
        const size_t batch_size = 1'000'000;
        std::vector<uint8_t> keys(batch_size * 32);
        std::vector<uint32_t> results(batch_size);
        
        while (!stop_flag) {
            // Przygotowanie danych
            for (size_t i = 0; i < batch_size; ++i) {
                Int key = random_key();
                memcpy(keys.data() + i*32, key.bits64, 32);
            }
            
            // Wysłanie na GPU
            clEnqueueWriteBuffer(gpu_queue, gpu_keys_buffer, CL_TRUE, 0,
                               batch_size * 32, keys.data(), 0, NULL, NULL);
            
            // Uruchomienie kernela
            size_t global_size = batch_size;
            clEnqueueNDRangeKernel(gpu_queue, gpu_kernel, 1, NULL,
                                  &global_size, NULL, 0, NULL, NULL);
            
            // Pobranie wyników
            clEnqueueReadBuffer(gpu_queue, gpu_results_buffer, CL_TRUE, 0,
                              batch_size * sizeof(uint32_t), results.data(),
                              0, NULL, NULL);
            
            // Przetworzenie wyników
            for (size_t i = 0; i < batch_size; ++i) {
                if (results[i]) {
                    Int found_key;
                    memcpy(found_key.bits64, keys.data() + i*32, 32);
                    report_found_key(found_key);
                }
            }
            
            keys_checked += batch_size;
        }
    }
#endif

public:
    void start_search(int cpu_threads_count) {
        // Inicjalizacja GPU
#ifdef WITH_GPU
        init_gpu();
        cpu_threads.emplace_back(&TurboSearchEngine::gpu_search_worker, this);
#endif

        // Uruchomienie wątków CPU
        for (int i = 0; i < cpu_threads_count; ++i) {
            cpu_threads.emplace_back(&TurboSearchEngine::cpu_search_worker, this, i);
        }
    }

    void stop_search() {
        stop_flag = true;
        for (auto& t : cpu_threads) {
            if (t.joinable()) t.join();
        }
    }
};