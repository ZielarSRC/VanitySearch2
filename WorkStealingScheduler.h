// WorkStealingScheduler.h
#pragma once
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>

class WorkStealingScheduler {
    std::vector<std::deque<std::function<void()>>> queues;
    std::vector<std::thread> workers;
    std::mutex global_mutex;
    std::condition_variable cv;
    bool stop = false;
    
public:
    WorkStealingScheduler(size_t num_threads) : queues(num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this, i] { worker_loop(i); });
        }
    }
    
    ~WorkStealingScheduler() {
        {
            std::lock_guard<std::mutex> lock(global_mutex);
            stop = true;
        }
        cv.notify_all();
        for (auto& t : workers) t.join();
    }
    
    void schedule(std::function<void()> task, size_t thread_idx = -1) {
        if (thread_idx == -1) thread_idx = rand() % queues.size();
        
        {
            std::lock_guard<std::mutex> lock(global_mutex);
            queues[thread_idx].push_back(std::move(task));
        }
        cv.notify_one();
    }
    
private:
    void worker_loop(size_t my_idx) {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(global_mutex);
                
                cv.wait(lock, [this, my_idx] {
                    return stop || !queues[my_idx].empty();
                });
                
                if (stop && queues[my_idx].empty()) return;
                
                if (!queues[my_idx].empty()) {
                    task = std::move(queues[my_idx].front());
                    queues[my_idx].pop_front();
                } else {
                    // Work stealing
                    for (size_t i = 0; i < queues.size(); ++i) {
                        if (i != my_idx && !queues[i].empty()) {
                            task = std::move(queues[i].back());
                            queues[i].pop_back();
                            break;
                        }
                    }
                }
            }
            
            if (task) task();
        }
    }
};