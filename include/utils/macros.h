#pragma once

#include <iostream>

//#define LIKELY(x) (__builtin_expect(!!(x), 1))
//#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#define LIKELY(x) (x) [[likely]]
#define UNLIKELY(x) (x) [[unlikely]]

inline auto FATAL(const std::string& msg) {
    std::cerr << "Fatal error: " << msg << std::endl;
    exit(EXIT_FAILURE);
}

inline auto ASSERT(bool condition, const std::string& msg) {
    if UNLIKELY(!condition) { 
        FATAL(msg);
    }
}