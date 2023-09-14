#pragma once

#include <atomic>
#include <vector>
#include <iostream>

// Single produser, single consumer, lock free queue.
template<typename T>
class SPSCQueue {
private:
    std::vector<T> storage_;
    std::atomic<size_t> write_idx_ = {0};
    std::atomic<size_t> read_idx_ = {0};
    std::atomic<size_t> size_ = {0};

public:
    SPSCQueue() = delete;
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue(const SPSCQueue&&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&&) = delete;

    SPSCQueue(size_t storage_size) 
    : storage_(storage_size, T()) {
    }

    size_t Size() const noexcept {
        return size_.load();
    }
    
    size_t Capacity() const noexcept {
        return storage_.size();
    }

    bool Read(T& value) noexcept {
        const auto v = ElementToRead();

        if (!v) {
            return false;
        }

        value = *v;
        UpdateReadIdx();
        return true; 
    }

    bool Write(const T& value) noexcept {
        auto v = ElementToWrite();

        if (!v) {
            return false;
        }

        *v = value;
        UpdateWriteIdx();
        return true;
    }
private:
    inline T* ElementToWrite() noexcept {
        return (storage_.size() == size_) ? nullptr : &storage_[write_idx_];
    }

    // do not call this if ElementToWrite() returns nullptr
    inline void UpdateWriteIdx() noexcept {
        write_idx_ = (write_idx_ + 1) % storage_.size();
        ++size_;
    }

    inline const T* ElementToRead() const noexcept {
        return (0 == size_) ? nullptr : &storage_[read_idx_];
    }    

    // do not call this if ElementToRead() returns nullptr
    inline void UpdateReadIdx() noexcept {
        read_idx_ = (read_idx_ + 1) % storage_.size();
        --size_;
    }
};