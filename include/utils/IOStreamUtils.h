#pragma once

#include <Eigen/Geometry>
#include <iostream>
#include <array>

template<typename T, size_t Nm>
std::ostream& operator << (std::ostream& output, const std::array<T, Nm>& arr) { 
    if (arr.empty()) {
    output << "[]";   
    return output;
    }
    
    int size = std::min<size_t>(Nm, 10);
    output << "[" << arr[0];
    for(int i = 1; i < size; ++i) {
    output << ", " << arr[i];
    }
    if (Nm > 10)
        output << ", ... ";
    
    output << "]";
    return output;            
}

template<typename T>
std::ostream& operator << (std::ostream& output, const std::vector<T>& arr) { 
    if (arr.empty()) {
    output << "[]";   
    return output;
    }
    
    int size = std::min<size_t>(arr.size(), 10);
    output << "[" << arr[0];
    for(int i = 1; i < size; ++i) {
    output << ", " << arr[i];
    }
    if (arr.size() > 10)
        output << ", ... ";
    
    output << "]";
    return output;            
}