#pragma once

#include <camera_calibration/FrameRemapper.h>

#include <opencv2/core.hpp>
#include <Eigen/Core>

#include <optional>
#include <queue>
#include <memory>

// TUM Monocular Visual Odometry Dataset reader
// https://cvg.cit.tum.de/data/datasets/mono-dataset?redirect=1

class TUM_DatasetReader {
public:
    struct MonoImage {
        uint64_t time_;
        cv::Mat frame_;
    };
    TUM_DatasetReader(const std::string& path);
    std::optional<MonoImage> GetRawImage();
    std::optional<MonoImage> GetImage();

    const std::array<double, 7>& GetRawCamParam() const {return raw_cam_param_;}
    const std::array<double, 5>& GetRectCamParam()  const {return rect_cam_param_;}
private:
    void LoadTimestamps();
    void LoadVignette();
    void LoadInvResponseFunction();
    void LoadCamera();

    void CorrectColor(cv::Mat& img, double exposure = 1.0);

    std::string path_;

    struct ImageData {
        double time_;
        std::string name_;
        double exposure_;
    };

    std::queue<ImageData> images_;
    cv::Mat vignette_mat_;
    std::unique_ptr<FrameRemapper<double>> remapper_;

    std::vector<double> inv_responce_;

    std::array<double, 7> raw_cam_param_;
    std::array<double, 5> rect_cam_param_;
};