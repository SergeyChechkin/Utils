#include "utils/image/ColorConvertion.h"
#include <opencv2/imgproc.hpp>

cv::Mat ConvertToGray(const cv::Mat& src) {
    cv::Mat gray;
    if (CV_8U == src.type()) {
        gray = src.clone();
    } else if (CV_8UC3 == src.type()) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        throw std::invalid_argument("Expecting RGB or grayscale image.");
    }

    return gray;
}

cv::Mat ConvertToColor(const cv::Mat& src) {
    cv::Mat color;
    if (CV_8U == src.type()) {
        cv::cvtColor(src, color, cv::COLOR_GRAY2BGR);
    } else if (CV_8UC3 == src.type()) {
        color = src.clone();
    } else {
        throw std::invalid_argument("Expecting RGB or grayscale image.");
    }
    
    return color;
}