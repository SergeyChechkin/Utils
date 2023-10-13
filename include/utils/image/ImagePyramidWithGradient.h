/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "utils/image/ImageWithGradient.h"
#include "utils/macros.h"

#include <utils/ImageUtils.h>
#include <opencv2/imgproc.hpp>

template<unsigned sobel_option>
struct ImagePyramidWithGradientT {
    ImagePyramidWithGradientT() {
    }

    bool Empty() const {
        return pyromid_.empty();
    }

    explicit ImagePyramidWithGradientT(const cv::Mat& gray_img, int pyr_level) {
        ASSERT(CV_8U == gray_img.type(), "Accept only gray image.");

        std::vector<cv::Mat> pyromid;
        cv::buildPyramid(gray_img, pyromid, pyr_level);

        pyromid_.resize(pyromid.size());
        for(size_t i = 0; i < pyromid.size(); ++i) {
            pyromid_[i] = ImageWithGradient(pyromid[i]);
        }
    }

    std::vector<ImageWithGradient> pyromid_;
};

using ImagePyramidWithGradient = ImagePyramidWithGradientT<SobleOptions::soblel_5x5>;