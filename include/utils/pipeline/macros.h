#pragma once

#include <iostream>


inline auto FATAL(const std::string& msg) {
    std::cerr << "Fatal error: " << msg << std::endl;
    exit(EXIT_FAILURE);
}

inline auto ASSERT(bool condition, const std::string& msg) {
    if (!condition, 0) [[unlikely]] { 
        FATAL(msg);
    }
}