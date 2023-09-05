#pragma once

#include <atomic>
#include <thread>
#include <iostream>

inline bool SetCoreId(int core_id) noexcept {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return 0 == pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}; 

template <typename T, typename... A>
inline std::unique_ptr<std::thread> ExecuteInThread(
    int core_id, 
    T&& func,
    A &&... args) noexcept {

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
        std::forward<T>(func)((std::forward<A>(args))...); 
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