/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "utils/image/ImageWithGradient.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

/// @brief KLT (Kanade-Lucas-Tomasi) tracker with gain correction 
/// For details see publication: 
/// S. K. Kim, D. Gallup, J. Frahm, M. Pollefeys, 
/// ”Joint Radiometric Calibration and Feature Tracking for an Adaptive Stereo System.”
/// Computer Vision and Image Understanding, Vol. 114, pp. 574-582, May, 2010.
class KLT {
public:
    /// @brief Constuctor 
    /// @param src_img - first image with gradients 
    /// @param dst_img - second image with gradients
    /// @param patch_size - square patch size 
    KLT(const ImageWithGradient& src_img, const ImageWithGradient& dst_img, size_t patch_size, size_t max_itr); 

    /// @brief Features tracking, constant exposure assumption 
    /// @param src_pnts - features on first image 
    /// @param dst_pnts - [in, out] initial guess, tracked features on second image
    void Track(
        const std::vector<cv::Point2f>& src_pnts, 
        std::vector<cv::Point2f>& dst_pnts) const;

    /// @brief Single features tracking, constant exposure assumption 
    /// @param src_pnt - feature on first image 
    /// @param dst_pnt - [in, out] initial guess, tracked feature on second image
    void TrackPoint(
        const cv::Point2f& src_pnt, 
        cv::Point2f& dst_pnt) const;

    /// @brief Feature tracking with exposure correction 
    /// @param src_pnts - features on first image 
    /// @param dst_pnts - [in, out] initial guess, tracked features on second image
    /// @param gain - relative gain value  
    void TrackGainInvariant(
        const std::vector<cv::Point2f>& src_pnts, 
        std::vector<cv::Point2f>& dst_pnts, 
        double& gain) const;

    /// @brief Feature tracking with exposure correction 
    /// @param src_pnts - features on first image 
    /// @param dst_pnts - initial guess on second image
    /// @return - relative gain     
    double ComputeGain(
        const std::vector<cv::Point2f>& src_pnts, 
        const std::vector<cv::Point2f>& dst_pnts) const;
private:
    inline bool CheckPointLocation(const cv::Point2f& pnt) const;
private:
    // supports grayscale image only 
    static constexpr int image_type_ = CV_8U;
    using PixelT = uint8_t; 

    const ImageWithGradient& src_img_;
    const ImageWithGradient& dst_img_;

    size_t patch_size_;
    size_t max_itr_;
    int center_offset_;
    size_t err_size_;

    cv::Point2i top_left_;
    cv::Point2i btm_rght_;

    static constexpr int border_ = 2;
    static constexpr float min_error_change_ = 0.05f;
    static constexpr float min_error_ = 0.1f;
};