#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Geometry>
#include <queue>
#include <optional>
#include <array>

// EuRoC datasets reader
// https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

class EuRoC_DatasetReader {
public:
    template<typename ScalarT>
    using Pose3 = Eigen::Transform<ScalarT, 3, Eigen::Isometry>;
    using Pose3d = Pose3<double>;

    struct StereoImage {
        uint64_t time_;
        cv::Mat frame_0_;           /// left
        cv::Mat frame_1_;           /// rught
    };

    struct GroundTruth {
        uint64_t time_;
        Eigen::Vector3f p_RS_R_;    /// position [m]
        Eigen::Quaternionf q_RS_;   /// rotation []
        Eigen::Vector3f v_RS_R_;    /// linear velosity [m s^-1] 
        Eigen::Vector3f b_w_RS_S_;  /// angular velosity [rad s^-1]
        Eigen::Vector3f b_a_RS_S_;  /// linear acceleration [m s^-2]
    };

public:
    EuRoC_DatasetReader(const std::string& path, bool image_only = false);
    std::optional<StereoImage> GetImage();
    std::optional<GroundTruth> GetGT();

    Eigen::Isometry3d GetExtrinsic_cam0() const {return cam0_.extr_;}
    Eigen::Isometry3d GetExtrinsic_cam1() const {return cam1_.extr_;}

    std::array<double, 8> GetIntrinsic_cam0() const;
    std::array<double, 8> GetIntrinsic_cam1() const;
private:
    void LoadImages();
    void LoadGroundTruth();
    
    struct CameraConfig {
        Eigen::Isometry3d extr_;
        float frame_rate_;
        std::vector<double> intrinsics_;
        std::vector<double> distortion_;
    };

    CameraConfig LoadCameraConfig(std::string sub_path);

    struct StereoImgNames {
        StereoImgNames() {}
        StereoImgNames(const std::string& frame_0, const std::string& frame_1, uint64_t time) 
            : frame_0_(frame_0), frame_1_(frame_1), time_(time) {}
        std::string frame_0_;
        std::string frame_1_;
        uint64_t time_;
    };

    std::string path_;
    std::queue<StereoImgNames> imgs_; 
    std::queue<GroundTruth> gts_; 
    CameraConfig cam0_;
    CameraConfig cam1_;
};