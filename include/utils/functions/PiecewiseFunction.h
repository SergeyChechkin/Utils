/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <vector>

/// @brief Inverse value computation with simplified Newton's method 
/// @tparam T - scalar type
/// @tparam UnaryFunction -function type
/// @param val - target value of the function 
/// @param f - function 
/// @param max_itr - iteration limit
/// @return - function input to get val
template <typename T, typename UnaryFunction>
T GeInverseValue(T val, UnaryFunction f, size_t max_itr = 20) {
    // Iterative solution without J computation (assumption: J close to I).
    T r = val;
    for(size_t i = 0; i < max_itr; ++i) {
        const T fr = f(r);
        const T error = val - fr;
        if(std::abs(error) < std::numeric_limits<T>::epsilon()) {
            break;
        }
        r += error;
    }
    return r;
}

/// @brief Inverse function datapoints generation
/// Assumpton: monotonic function  
/// @tparam T - scalar type
/// @tparam UnaryFunction - function type
/// @param size - number of intervals 
/// @param min_fx - begin of the function range
/// @param max_fx - end of the function range
/// @param f - function
/// @return - datapoints
template <typename T, typename UnaryFunction>
std::vector<T> GeneratePiecewiseInverseFunction(size_t size, T min_fx, T max_fx, UnaryFunction f) {
    std::vector<T> data(size);

    T scalse = (max_fx - min_fx) / size;
    T prev = f(min_fx);
    for(size_t i = 1; i < size; ++i) {
        T next = f(min_fx + scalse * i);
        T idx = std::ceil(prev);
        T intrvl = (next - prev);
        if (intrvl > std::numeric_limits<T>::epsilon()) { 
            while(idx <= next) {
                data[idx] = (i-1) + (idx - prev) / intrvl;
                ++idx;
            }
        }

        prev = next; 
    }

    return data;
}

/// @brief Fixed size interval piecewise function with linear interpolation
/// @tparam T - scalar type
template <typename T>
class PiecewiseFunction {
public:
    PiecewiseFunction(const std::vector<T>& data) : data_(data), min_x_(0), max_x_(data.size() - 1), inv_scale_(1) {
    }

    PiecewiseFunction(const std::vector<T>& data, T min_x, T max_x) : data_(data), min_x_(min_x), max_x_(max_x) {
        inv_scale_ = data.size() / (max_x_ - min_x_);
    }
    
    T operator()(const T& x) const {
        if (x <= min_x_) {
            return data_.front();
        } else if (x >= max_x_) {
            return data_.back();
        } else {
            T idx_x = (x - min_x_) * inv_scale_;
            T idx_bx = std::floor(idx_x);
            T v_beging = data_[idx_bx];
            T v_end = data_[idx_bx + 1];
            return (v_end - v_beging) * (idx_x - idx_bx) + v_beging;   
        }
    }
private:
    std::vector<T> data_;
    T min_x_;
    T max_x_;
    T inv_scale_; 
};

