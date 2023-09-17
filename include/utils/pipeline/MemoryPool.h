#pragma once

#include "macros.h"
#include <vector>
#include <iostream>

template<typename T>
class MemoryPool {
public:
    MemoryPool() = delete;
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool(const MemoryPool&&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&&) = delete;

    MemoryPool(size_t size) 
    : storage_(size, {T(), true})
    , next_free_block_idx_(0)
    , size_(0) {
    }

    inline bool IsFull() const noexcept {
        return size_ < storage_.size();
    }

    inline size_t Size() const noexcept {
        return size_;
    }

    inline size_t Capasity() const noexcept {
        return storage_.size();
    }

    template<typename...Args>
    T* allocate(Args... args) noexcept {
        if (!storage_[next_free_block_idx_].is_free_) {
            if (!FindNextFreeBlock()) {
                return nullptr;
            }
        } 
        auto block = &(storage_[next_free_block_idx_]);
        T* result = &(block->object_);
        result = new(result) T(args...);
        //*result = T(args...);
        block->is_free_ = false;
        ++size_;

        return result;
    }

    void deallocate(const T* v) noexcept {
        const auto idx = (reinterpret_cast<const Block*>(v) - &(storage_[0]));  
        ASSERT(idx >= 0 && idx < storage_.size(), "Element doesn't belong to this pool.");
        
        if UNLIKLY(!storage_[idx].is_free_) {
            storage_[idx].is_free_ = true;
            --size_;
        }
        next_free_block_idx_ = idx; // TODO: make it optional for different strategy ?
    }
private:
    bool FindNextFreeBlock() noexcept {
        if(size_ == storage_.size())
            return false;

        const auto initial_idx = next_free_block_idx_;
        for(; next_free_block_idx_ < storage_.size(); ++next_free_block_idx_) {
            if (storage_[next_free_block_idx_].is_free_) {
                return true;
            }
        }

        next_free_block_idx_ = 0;

        for(; next_free_block_idx_ < initial_idx; ++next_free_block_idx_) {
            if (storage_[next_free_block_idx_].is_free_) {
                return true;
            }
        }

        return false; // shouldn't get here. 
    }
private:
    struct Block {
        T object_;
        bool is_free_ = true;
    };

    std::vector<Block> storage_;
    size_t next_free_block_idx_;
    size_t size_;
};