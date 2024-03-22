/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <utils/features/MinFeature/MinFeature.h>
#include <utils/image/ImageWithGradient.h>
#include <Eigen/Core>

namespace lib::features {

class MinFeaturesExtractor {
public:
    struct Configuration {
        float threshold_ = 8;   // based on image noise 
        int square_size_ = 32;  // for even distribution 
        int square_count_ = 16; // for even distribution 
    };
public:
    static std::vector<MinFeature2D> Extract(
        const ImageWithGradient& image, 
        const Configuration& config);   
    
    static void SubPixelLocation(
        const ImageWithGradient& image,
        MinFeature2D& feature);

    static cv::Mat DisplayFeatures(
        const cv::Mat& image, 
        const std::vector<MinFeature2D>& features, 
        cv::Vec3b color = {0, 0, 255});
};

}