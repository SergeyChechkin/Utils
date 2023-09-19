#include <utils/dataset_reader/EuRoC_DatasetReader.h>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Geometry>

#include <fstream>
#include <iostream>

EuRoC_DatasetReader::EuRoC_DatasetReader(const std::string& path, bool image_only) : path_(path) {
    LoadImages();

    if (!image_only) {
        LoadGroundTruth();
        cam0_ = LoadCameraConfig("/mav0/cam0/sensor.yaml");
        cam1_ = LoadCameraConfig("/mav0/cam1/sensor.yaml");
    }
}

void EuRoC_DatasetReader::LoadImages() {
    std::string cam0_file_name = path_ + "/mav0/cam0/data.csv";
    std::string cam1_file_name = path_ + "/mav0/cam1/data.csv";
    
    std::ifstream cam0_file (cam0_file_name);
    std::ifstream cam1_file (cam1_file_name);
    
    if (cam0_file.is_open() && cam1_file.is_open()) {
        std::string line0, line1;
        std::getline(cam0_file, line0);
        std::getline(cam1_file, line1);

        while(std::getline(cam0_file, line0) && std::getline(cam1_file, line1))
		{
			std::stringstream str0(line0);
            std::stringstream str1(line1);
 
            uint64_t time_0, time_1;
            char c;
            std::string file_0, file_1;

            str0 >> time_0 >> c >> file_0;
            str1 >> time_1 >> c >> file_1;
            
            if (time_0 == time_1) {
               imgs_.emplace(file_0, file_1, time_0);
            }
		}
    }  
}


std::optional<EuRoC_DatasetReader::StereoImage> EuRoC_DatasetReader::GetImage() {
    if (imgs_.empty()) 
        return {};

    const StereoImgNames& img = imgs_.front();
    
    StereoImage result;
    result.frame_0_ = cv::imread(path_ + "/mav0/cam0/data/" + img.frame_0_);
    result.frame_1_ = cv::imread(path_ + "/mav0/cam1/data/" + img.frame_1_);
    result.time_ = img.time_;
    imgs_.pop();
    return result;
}

void EuRoC_DatasetReader::LoadGroundTruth() {
    std::string gt_file_name = path_ + "/mav0/state_groundtruth_estimate0/data.csv";  
    std::ifstream gt_file(gt_file_name);

    if (gt_file.is_open()) {
        std::string line;
        // skip column headers
        std::getline(gt_file, line);

        while(std::getline(gt_file, line))
		{
			std::stringstream ss(line);
            GroundTruth gt;
            char c;
            ss >> gt.time_ >> c;
            ss >> gt.p_RS_R_.x() >> c >> gt.p_RS_R_.y() >> c >> gt.p_RS_R_.z() >> c;
            ss >> gt.q_RS_.w() >> c >> gt.q_RS_.x() >> c >> gt.q_RS_.y() >> c >> gt.q_RS_.z() >> c;
            ss >> gt.v_RS_R_.x() >> c >> gt.v_RS_R_.y() >> c >> gt.v_RS_R_.z() >> c;
            ss >> gt.b_w_RS_S_.x() >> c >> gt.b_w_RS_S_.y() >> c >> gt.b_w_RS_S_.z() >> c;
            ss >> gt.b_a_RS_S_.x() >> c >> gt.b_a_RS_S_.y() >> c >> gt.b_a_RS_S_.z();
            gts_.push(gt);
		}
    }
}

std::optional<EuRoC_DatasetReader::GroundTruth> EuRoC_DatasetReader::GetGT() {
    if (gts_.empty()) 
        return {};
    
    GroundTruth gt = gts_.front();
    gts_.pop();
    return gt;
}  

EuRoC_DatasetReader::CameraConfig EuRoC_DatasetReader::LoadCameraConfig(std::string sub_path) {
    CameraConfig result;
    std::string gt_conf_file_name = path_ + sub_path; 
    YAML::Node config = YAML::LoadFile(gt_conf_file_name);
    YAML::Node extr_node = config["T_BS"];
    Eigen::Matrix4d Matrix4x4 = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(extr_node["data"].as<std::vector<double>>().data());
    result.extr_.matrix() = Matrix4x4;
    result.frame_rate_ = config["rate_hz"].as<float>();
    result.intrinsics_ = config["intrinsics"].as<std::vector<double>>();
    result.distortion_ = config["distortion_coefficients"].as<std::vector<double>>();
    return result;
}

std::array<double, 8> EuRoC_DatasetReader::GetIntrinsic_cam0() const {
    std::array<double, 8> result;

    for(int i = 0; i < 4; ++i)  {
        result[i] = cam0_.intrinsics_[i];
        result[i+4] = cam0_.distortion_[i];
    }

    return result;
}

std::array<double, 8> EuRoC_DatasetReader::GetIntrinsic_cam1() const {
    std::array<double, 8> result;

    for(int i = 0; i < 4; ++i)  {
        result[i] = cam1_.intrinsics_[i];
        result[i+4] = cam1_.distortion_[i];
    }

    return result;
}