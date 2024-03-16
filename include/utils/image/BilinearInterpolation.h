/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <opencv2/core.hpp>
#include <optional>

/// @brief Bilinear interpolation on image with border check
/// @tparam ScalarT - float or double 
/// @tparam ImageT - float for CV_32F, uint8 for CV_8U 
/// @param image - image 
/// @param point - point 
/// @return - interpolated value if sucsesfull
template<typename ScalarT, typename ImageT = ScalarT>
std::optional<ScalarT> BilinearValue(const cv::Mat& image, const ScalarT point[2]) {
    const ScalarT u = point[0]; 
    const ScalarT v = point[1];
    
    const int left = static_cast<int>(std::floor(u));  
    const int top = static_cast<int>(std::floor(v));

    if(left < 0 || top < 0 || left + 1 >= image.cols || top + 1 >= image.rows) {
        return {};
    }

    const ScalarT x = u - left;
    const ScalarT y = v - top;
    const ScalarT x_1 = 1 - x;
    const ScalarT y_1 = 1 - y;

    const ImageT value_0_0 = image.at<ImageT>(top, left);
    const ImageT value_0_1 = image.at<ImageT>(top, left + 1);
    const ImageT value_1_0 = image.at<ImageT>(top + 1, left);
    const ImageT value_1_1 = image.at<ImageT>(top + 1, left + 1);

    return value_0_0 * x_1 * y_1 + value_0_1 * x * y_1 + value_1_0 * x_1 * y + value_1_1 * x * y;
}

/// @brief Bilinear interpolation on image without border check
/// @tparam ScalarT - float or double 
/// @tparam ImageT - float for CV_32F, uint8 for CV_8U 
/// @param image - image 
/// @param point - point 
/// @return - interpolated value
template<typename ScalarT, typename ImageT = ScalarT>
ScalarT BilinearValue_(const cv::Mat& image, const ScalarT point[2]) {
    const ScalarT u = point[0];
    const ScalarT v = point[1];
    
    const int left = static_cast<int>(std::floor(u));  
    const int top = static_cast<int>(std::floor(v));

    const ScalarT x = u - left;
    const ScalarT y = v - top;
    const ScalarT x_1 = 1 - x;
    const ScalarT y_1 = 1 - y;

    const ImageT value_0_0 = image.at<ImageT>(top, left);
    const ImageT value_0_1 = image.at<ImageT>(top, left + 1);
    const ImageT value_1_0 = image.at<ImageT>(top + 1, left);
    const ImageT value_1_1 = image.at<ImageT>(top + 1, left + 1);

    return value_0_0 * x_1 * y_1 + value_0_1 * x * y_1 + value_1_0 * x_1 * y + value_1_1 * x * y;
}

template<typename ScalarT, typename ImageT = ScalarT>
bool BilinearImageCheck(const cv::Mat& image, const ScalarT point[2]) {
    const ScalarT u = point[0]; 
    const ScalarT v = point[1];
    
    const int left = static_cast<int>(std::floor(u));  
    const int top = static_cast<int>(std::floor(v));

    return !(left < 0 || top < 0 || left + 1 >= image.cols || top + 1 >= image.rows);
}