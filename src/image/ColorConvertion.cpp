#include "utils/image/ColorConvertion.h"
#include <utils/macros.h>
#include <opencv2/imgproc.hpp>

cv::Mat ConvertToGray(const cv::Mat& src) {
    cv::Mat gray;
    if (CV_8U == src.type()) {
        gray = src.clone();
    } else if (CV_8UC3 == src.type()) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else [[unlikely]] {
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
    } else [[unlikely]] {
        throw std::invalid_argument("Expecting RGB or grayscale image.");
    }
    
    return color;
}

cv::Mat ImageCat(const cv::Mat& first, const cv::Mat& second) {
    ASSERT(first.type() == second.type(), "Expect same type images.");
    
    cv::Size res_size(first.cols + second.cols, std::max(first.rows, second.rows)); 
    cv::Mat result(res_size.height, res_size.width, first.type(), cv::Scalar(0));
    cv::Mat first_part(result, cv::Range(0, first.rows), cv::Range(0, first.cols));
    first.copyTo(first_part);
    cv::Mat second_part(result, cv::Range(0, second.rows), cv::Range(first.cols, result.cols));
    second.copyTo(second_part);

    return result;
}
