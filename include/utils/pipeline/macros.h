#pragma once

#include <iostream>

//#define LIKLY(x) (__builtin_expect(!!(x), 1))
//#define UNLIKLY(x) (__builtin_expect(!!(x), 0))
#define LIKLY(x) (x) [[likely]]
#define UNLIKLY(x) (x) [[unlikely]]

inline auto FATAL(const std::string& msg) {
    std::cerr << "Fatal error: " << msg << std::endl;
    exit(EXIT_FAILURE);
}

inline auto ASSERT(bool condition, const std::string& msg) {
    if UNLIKLY(!condition) { 
        FATAL(msg);
    }
}