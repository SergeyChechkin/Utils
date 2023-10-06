/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <opencv2/core.hpp>

// color convertion
cv::Mat ConvertToGray(const cv::Mat& src);
cv::Mat ConvertToColor(const cv::Mat& src);

// concatenate images
cv::Mat ImageCat(const cv::Mat& first, const cv::Mat& second);