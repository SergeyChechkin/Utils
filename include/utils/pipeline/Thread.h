#pragma once

#include <utils/macros.h>

#include <atomic>
#include <thread>
#include <iostream>

inline bool SetCoreId(int core_id) noexcept {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return 0 == pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}; 

template <typename T, typename... Args>
inline std::unique_ptr<std::thread> ExecuteInThread(
    int core_id, 
    T&& func,
    Args &&... args) noexcept {

    std::atomic<bool> failed(false);
    std::atomic<bool> running(false);
    
    auto thread_body = [&] {
        if (core_id >= 0 && !SetCoreId(core_id)) {
            std::cerr << "Faild to set core id for "  
                << pthread_self() << " to " << core_id << "." << std::endl;  
            failed = true;
            return;
        }

        running = true;
        std::forward<T>(func)((std::forward<Args>(args))...); 
    };

    auto thread = std::make_unique<std::thread>(thread_body);

    while(!running && !failed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } 

    if (failed) {
        thread->join();
        return {};
    }

    return thread;
}

void ParallelFor(
    size_t begin, 
    size_t end, 
    const std::function<void(size_t begin, size_t end)>& job, 
    size_t num_threads = std::thread::hardware_concurrency()) {
        
        const size_t size = end - begin;

        if UNLIKELY(num_threads < 2 || size < 2) {
            job(begin, end);
            return; 
        }

        size_t interval = size / num_threads;
        size_t residual = size % num_threads;

        if UNLIKELY(interval < 1) {
            num_threads = size;
            interval = 1;
            residual = 0;
        }

        size_t job_begin = begin;
        size_t job_end;

        std::vector<std::thread> threads;

        while(job_begin < end) {
            job_end = std::min(end, job_begin + interval);
            if (residual > 0) {
                ++job_end;
                --residual;
            }
            
            threads.push_back(std::thread(job, job_begin, job_end));
            
            job_begin = job_end;
        }

        for(size_t i = 0; i < num_threads; ++i) {
            threads[i].join();
        }
}