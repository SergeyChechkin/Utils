/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <utils/ImageUtils.h>
#include <opencv2/imgproc.hpp>

struct SobleOptions {
    static constexpr unsigned soblel_3x3 = 0;
    static constexpr unsigned soblel_5x5 = 1;
};

/// @brief Grayscale image with gradient  
/// @tparam sobel_option - Sobel filter configuration
template<unsigned sobel_option>
struct ImageWithGradientT {

    ImageWithGradientT() {
    }

    bool Empty() const  {
        return gray_.empty();
    }

    explicit ImageWithGradientT(const cv::Mat& src) {
        gray_ = ConvertToGray(src);
        
        if constexpr(sobel_option == SobleOptions::soblel_5x5) {
            cv::Sobel(gray_, grad_x_, CV_32F, 1, 0, 5, 1.0 / 128.0);
            cv::Sobel(gray_, grad_y_, CV_32F, 0, 1, 5, 1.0 / 128.0);
        } else if constexpr(sobel_option == SobleOptions::soblel_3x3) {
            cv::Sobel(gray_, grad_x_, CV_32F, 1, 0, 3, 1.0 / 8.0);
            cv::Sobel(gray_, grad_y_, CV_32F, 0, 1, 3, 1.0 / 8.0);
        }
    }

    ImageWithGradientT<sobel_option> GetPatchSubPix(const cv::Size& patch_size, const cv::Point2f& center) const {
        ImageWithGradientT<sobel_option> result;

        /// cv::getRectSubPix for patch point outside the image populate with edge values
        cv::getRectSubPix(gray_, patch_size, center, result.gray_, CV_32F);
        cv::getRectSubPix(grad_x_, patch_size, center, result.grad_x_, CV_32F);
        cv::getRectSubPix(grad_y_, patch_size, center, result.grad_y_, CV_32F);
        
        return result;
    }

    cv::Mat gray_;
    cv::Mat grad_x_;
    cv::Mat grad_y_;
};

using ImageWithGradient = ImageWithGradientT<SobleOptions::soblel_5x5>;