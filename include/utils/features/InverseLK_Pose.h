#pragma once

#include <camera_model/PinholeCameraModel.h>
#include <utils/image/ImageWithGradient.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// Feature tracker with rigid body transformation constraint.
// Inverse compositional image alignment
class InverseLK_Pose {
public:
    using CameraModelT = PinholeCameraModel<double>;
    InverseLK_Pose(const CameraModelT& cm, const ImageWithGradient& src_img, const ImageWithGradient& dst_img); 
    bool Track(
        const std::vector<Eigen::Vector2d>& src_img_pnts,
        const std::vector<Eigen::Vector3d>& src_obj_pnts, 
        Eigen::Isometry3d& pose) const;
private:
    static Eigen::Vector2d PatchPointFromIndex(int idx);
    bool CheckPointLocation(const Eigen::Vector2d& pnt) const;

private:
    // grayscale image only, just in case if need to implement for float image in the future 
    static constexpr int image_type_ = CV_8U;
    using PixelT = uint8_t; 

    CameraModelT cm_;
    double f_;
    ImageWithGradient src_img_;
    ImageWithGradient dst_img_;
    
    static constexpr int border_ = 2;
    static constexpr int patch_size_ = 5;
    static constexpr int center_offset_ = patch_size_ / 2;
    static constexpr int err_size_ = patch_size_ * patch_size_;
    
    const size_t max_itr_ = 20;
    const double min_error_change_ = 0.1;
    const double min_error_ = 0.5;  // average intensity difference per pixel

    Eigen::Vector2d top_left_;
    Eigen::Vector2d btm_rght_;
};