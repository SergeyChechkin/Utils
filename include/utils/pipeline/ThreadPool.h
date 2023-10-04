#pragma once

#include <utils/macros.h>

#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <queue>
#include <iostream>

class ThreadPool {
private:
    std::queue<std::function<void()>> jobs_;
    std::mutex queue_mutex_;
    std::condition_variable mutex_condition_;
    std::vector<std::thread> threads_;
    std::atomic<bool> terminate_ = {false};
public:
    void ParallelFor(
        size_t begin, 
        size_t end, 
        const std::function<void(size_t begin, size_t end)>& job, 
        size_t num_threads = std::thread::hardware_concurrency()) noexcept 
    {
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

        std::counting_semaphore semaphore_(0); 

        auto thread_body = [&](size_t begin, size_t end) {
            job(begin, end);
            semaphore_.release();
        };


        while(job_begin < end) {
            job_end = std::min(end, job_begin + interval);
            if (residual > 0) {
                ++job_end;
                --residual;
            }
            QueueJob([=]{thread_body(job_begin, job_end);});
            job_begin = job_end;
        }

        for(size_t i = 0; i < num_threads; ++i) {
            semaphore_.acquire();
        }
    }

public:
    void Start(size_t num_threads = std::thread::hardware_concurrency()) noexcept {
        ASSERT(num_threads > 0, "Invalid number of threads.");
        terminate_ = false;
        threads_.reserve(num_threads);
        for(unsigned int i = 0; i < num_threads; ++i) {
            threads_.emplace_back(std::thread(&ThreadPool::ThreadLoop, this));
        }
    }

    // use case: thrad_pool.QueueJob([]{/* ... */});
    void QueueJob(const std::function<void()>& job) noexcept {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            jobs_.push(job);
        }

        mutex_condition_.notify_one();
    }

    void Stop() noexcept {
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

    bool Busy() noexcept {
        bool result;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            result = !jobs_.empty();
        }

        return result;
    }
private:
    void ThreadLoop() noexcept {
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
};