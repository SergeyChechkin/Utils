#include <utils/dataset_reader/TUM_DatasetReader.h>

#include <camera_model/GeometricCameraModel.h>
#include <utils/IOStreamUtils.h>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>

TUM_DatasetReader::TUM_DatasetReader(const std::string& path)
: path_(path)
{
    LoadTimestamps();
    LoadVignette();
    LoadInvResponseFunction();
    LoadCamera();
} 

void TUM_DatasetReader::LoadTimestamps()
{
    std::string file_name = path_ + "/times.txt";  
    std::ifstream file(file_name);

    if (!file.is_open())
        return; 

    std::string line;
    while(std::getline(file, line))
    {
        std::stringstream ss(line);
        ImageData idata;
        char c;
        ss >> idata.name_ >> c >> idata.time_ >> c >> idata.exposure_;
        images_.push(idata);
    }

}

void TUM_DatasetReader::LoadVignette() 
{
    cv::Mat vignette_img = cv::imread(path_ + "/vignette.png", cv::IMREAD_UNCHANGED);
    vignette_img.convertTo(vignette_mat_, CV_32F, 1.0 / (254.9 * 254.9));
    
//    std::cout << vignette_img.type() << std::endl;
//    cv::imshow("vignette_img", vignette_img);
//    cv::waitKey();
}

void TUM_DatasetReader::LoadInvResponseFunction()
{
    std::string file_name = path_ + "/pcalib.txt";  
    std::ifstream file(file_name);

    if (!file.is_open())
        return; 
    
    std::string line;
    if (!std::getline(file, line))
        return;

    inv_responce_.resize(256);

    std::stringstream ss(line);
    for(int i = 0; i < 256; ++i) {
        ss >> inv_responce_[i];
    } 
}

void TUM_DatasetReader::LoadCamera()
{
    std::string file_name = path_ + "/camera.txt";  
    std::ifstream file(file_name);
    if (!file.is_open())
        return; 

    /// src: FOV model
    double fx, fy, cx, cy, w;
    int width, height;

    {
        std::string line_1, line_2;
        if (!std::getline(file, line_1) || !std::getline(file, line_2))
            return;

        std::stringstream ss1(line_1);
        ss1 >> fx >> fy >> cx >> cy >> w;

        std::stringstream ss2(line_2);
        ss2 >> width >> height;

        fx *= width;
        fy *= height;

        cx *= width;
        cy *= height;
    }

    std::shared_ptr<FieldOfView<double>> src_pojrction = std::make_shared<FieldOfView<double>>(0.5 * (fx + fy), w);
    std::shared_ptr<NullDistortion<double>> distortion = std::make_shared<NullDistortion<double>>();
    GeometricCameraModel<double> src(src_pojrction, distortion, {cx, cy}, {width, height});
    raw_cam_param_ = {fx, fy, cx, cy, w, double(width), double(height)};

    {
        std::string line_1, line_2;
        if (!std::getline(file, line_1) || !std::getline(file, line_2))
            return;

        std::stringstream ss1(line_1);
        ss1 >> fx >> fy >> cx >> cy;

        std::stringstream ss2(line_2);
        ss2 >> width >> height;

        fx *= width;
        fy *= height;

        cx *= width;
        cy *= height;
    }

    double f = 0.5 * (fx + fy);
    std::shared_ptr<Perspective<double>> dst_pojrction = std::make_shared<Perspective<double>>(f);
    GeometricCameraModel<double> dst(dst_pojrction, distortion, {cx, cy}, {width, height});
    rect_cam_param_ = {f, cx, cy, double(width), double(height)};

    remapper_ = std::make_unique<FrameRemapper<double>>(src, dst, Eigen::Matrix3d::Identity());
}


std::optional<TUM_DatasetReader::MonoImage> TUM_DatasetReader::GetRawImage() 
{
    if (images_.empty()) 
        return {};

    const auto& img = images_.front();
    
    MonoImage result;
    result.frame_ = cv::imread(path_ + "/images/" + img.name_ + ".jpg");
    result.time_ = img.time_;
    images_.pop();
    return result;
}

std::optional<TUM_DatasetReader::MonoImage> TUM_DatasetReader::GetImage()
{
    if (images_.empty()) 
        return {};

        const auto& img = images_.front();

    MonoImage result;
    result.frame_ = cv::imread(path_ + "/images/" + img.name_ + ".jpg", cv::IMREAD_GRAYSCALE);
    result.time_ = img.time_;
    images_.pop();

    CorrectColor(result.frame_);
    result.frame_ = remapper_->Remap(result.frame_);

    return result;
}

void TUM_DatasetReader::CorrectColor(cv::Mat& img, double exposure)
{
    for(int v = 0; v < img.rows; ++v) {
        uchar* row = img.ptr<uchar>(v);
        for(int u = 0; u < img.cols; ++u) {
            uchar val = row[u]; 
            const double inv_resp_value = inv_responce_.at(val);
            const double veg_value = vignette_mat_.at<float>(v, u);
            const double corect_val = inv_resp_value / (veg_value * exposure);
            row[u] = std::max<double>(std::min<double>(corect_val, 255), 0);
        }    
    }    
}
