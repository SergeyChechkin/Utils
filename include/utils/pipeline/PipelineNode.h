#pragma once

#include "SPSCQueue.h"
#include "Thread.h"
#include <atomic>
#include <memory>

enum NodeStatus {
    stop,
    runing,
    done,
};

template<typename T>
struct QueueWithStatus {
    QueueWithStatus(SPSCQueue<T>* queue, std::atomic<NodeStatus>* status) : queue_(queue), status_(status) {}
    SPSCQueue<T>* queue_;
    std::atomic<NodeStatus>* status_;
};

template<typename in_T, typename out_T>
class PipelineNode {
public:
    PipelineNode(std::shared_ptr<QueueWithStatus<out_T>> in_queue, size_t out_queue_size) 
    : in_queue_(in_queue)
    , out_queue_(out_queue_size)
    , status_(NodeStatus::stop)
    { };

    template<typename T, typename... Args>
    void Start(T&& func, Args &&... args) noexcept {
        auto thread_body = [&] {
            status_ = NodeStatus::runing;
            std::forward<T>(func)(in_queue_.get(), &out_queue_, (std::forward<Args>(args))...);
            status_ = NodeStatus::done;
        };

        thread_ = std::make_unique<std::thread>(thread_body);
    }

    void Join() noexcept {
        thread_->join();
    }

    std::shared_ptr<QueueWithStatus<out_T>> GetOutQueue() noexcept {
        return std::make_shared<QueueWithStatus<out_T>>(&out_queue_, &status_);
    }

private:
    std::shared_ptr<QueueWithStatus<out_T>> in_queue_;
    SPSCQueue<out_T> out_queue_;
    std::unique_ptr<std::thread> thread_;
    std::atomic<NodeStatus> status_;
};