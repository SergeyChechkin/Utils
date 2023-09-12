#pragma once

#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <queue>
#include <iostream>

class ThreadPool {
public:
    void Start() {
        terminate_ = false;
        const unsigned int num_threads = std::thread::hardware_concurrency();
        threads_.reserve(num_threads);
        for(unsigned int i = 0; i < num_threads; ++i) {
            threads_.emplace_back(std::thread(&ThreadPool::ThreadLoop, this));
        }
    }

    // use case: thrad_pool.QueueJob([]{/* ... */});
    void QueueJob(const std::function<void()>& job) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            jobs_.push(job);
        }

        mutex_condition_.notify_one();
    }

    void Stop() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            terminate_ = true;
        }

        mutex_condition_.notify_all();

        for(auto& thread : threads_) {
            thread.join();
        }

        threads_.clear();
    }

    bool Busy() {
        bool result;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            result = !jobs_.empty();
        }

        return result;
    }
private:
    void ThreadLoop() {
        while(true) {

            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                mutex_condition_.wait(lock, [this] {
                    return !jobs_.empty() || terminate_;
                });

                if (terminate_)
                    return;

                job = jobs_.front();
                jobs_.pop(); 
            }

            job();
        }
    }
private:
    std::queue<std::function<void()>> jobs_;
    std::mutex queue_mutex_;
    std::condition_variable mutex_condition_;
    std::vector<std::thread> threads_;
    std::atomic<bool> terminate_;
};