#include "utils/features/MinFeature/MinFeatureExtractor.h"
#include <spatial_hash/SpatialHash2DVector.h>
#include <spatial_hash/SpatialHash2DHeap.h>
#include <map>

using namespace lib::features;
using namespace libs::spatial_hash;

std::vector<MinFeature2D> MinFeaturesExtractor::Extract(
        const ImageWithGradient& image, 
        const Configuration& config) 
    {
    
    // we use gradient norm square as a feature function 
    cv::Mat sqr_grad_x, sqr_grad_y;
    cv::multiply(image.grad_x_, image.grad_x_, sqr_grad_x);
    cv::multiply(image.grad_y_, image.grad_y_, sqr_grad_y);
    cv::Mat grad_norm_sqr = sqr_grad_x + sqr_grad_y;

    // Dilate in order to highlight local max
    cv::Mat dilate_grad_norm_sqr;
    cv::dilate(grad_norm_sqr, dilate_grad_norm_sqr, cv::Mat(), cv::Point(-1,-1), config.dilate_size_);

    const int height = dilate_grad_norm_sqr.rows;
    const int width = dilate_grad_norm_sqr.cols;
    const int offset = image.PatchSize() / 2;
    const int result_size = height * width * config.square_count_ / (config.square_size_ * config.square_count_);
    // In order to have evenly distributed features
    // we limit number of features in each sector        

    SpatialHashTable2DHeap<float, float, MinFeature2D> hash_table_(config.square_size_, config.square_count_); 

    for(int v = offset; v < height - offset; ++v) {
        const float* dI_norm_row = grad_norm_sqr.ptr<float>(v);
        const float* dI_norm_dilate_row = dilate_grad_norm_sqr.ptr<float>(v);
        const float* dIx_row = image.grad_x_.ptr<float>(v);
        const float* dIy_row = image.grad_y_.ptr<float>(v);
        
        for(int u = offset; u < width - offset; ++u) {
            const float dI_norm = dI_norm_row[u];
            const float dI_norm_dilate = dI_norm_dilate_row[u];
            
            if (dI_norm == dI_norm_dilate && dI_norm > config.threshold_) {
                MinFeature2D feature;
                feature.location_ << u, v;  
                feature.gradient_ << dIx_row[u], dIy_row[u];
                //feature.responce_ = dI_norm;
                hash_table_.Add(feature.location_.data(), dI_norm, feature);
            }            
        }
    }

    return hash_table_.GetAllData();
}

std::vector<MinFeature2D> MinFeaturesExtractor::ExtractAll(
    const ImageWithGradient& image, 
    const Configuration& config)
{
    // we use gradient norm square as a feature function 
    cv::Mat sqr_grad_x, sqr_grad_y;
    cv::multiply(image.grad_x_, image.grad_x_, sqr_grad_x);
    cv::multiply(image.grad_y_, image.grad_y_, sqr_grad_y);
    cv::Mat grad_norm_sqr = sqr_grad_x + sqr_grad_y;

    // Dilate in order to highlight local max
    cv::Mat dilate_grad_norm_sqr;
    cv::dilate(grad_norm_sqr, dilate_grad_norm_sqr, cv::Mat(), cv::Point(-1,-1), config.dilate_size_);

    const int height = dilate_grad_norm_sqr.rows;
    const int width = dilate_grad_norm_sqr.cols;
    const int offset = image.PatchSize() / 2;
    const int result_size = height * width / (config.dilate_size_ * config.dilate_size_);
    
    std::vector<MinFeature2D> result;
    result.reserve(result_size);

    for(int v = offset; v < height - offset; ++v) {
        const float* dI_norm_row = grad_norm_sqr.ptr<float>(v);
        const float* dI_norm_dilate_row = dilate_grad_norm_sqr.ptr<float>(v);
        const float* dIx_row = image.grad_x_.ptr<float>(v);
        const float* dIy_row = image.grad_y_.ptr<float>(v);
        
        for(int u = offset; u < width - offset; ++u) {
            const float dI_norm = dI_norm_row[u];
            const float dI_norm_dilate = dI_norm_dilate_row[u];
            
            if (dI_norm == dI_norm_dilate && dI_norm > config.threshold_) {
                MinFeature2D feature;
                feature.location_ << u, v;  
                feature.gradient_ << dIx_row[u], dIy_row[u];
                //feature.responce_ = dI_norm;
                result.push_back(feature);
            }            
        }
    }

    return result;
}   


void MinFeaturesExtractor::SubPixelLocation(
    const ImageWithGradient& image,
    MinFeature2D& feature) 
{
    // Square polinomiial interpolation along gradient direction, max location extraction 
    const auto unit_grad = feature.gradient_.normalized();

    const Eigen::Vector2f p0 = feature.location_ - unit_grad;
    const Eigen::Vector2f p1 = feature.location_;
    const Eigen::Vector2f p2 = feature.location_ + unit_grad;

    const float v0 = image.GetSubPixGradient(p0.data()).norm();
    const float v1 = image.GetSubPixGradient(p1.data()).norm();
    const float v2 = image.GetSubPixGradient(p2.data()).norm();

    // Square polynomial interpolation 
    const float dx_01 = v1 - v0;
    const float dx_12 = v2 - v1;
    const float x = 0.5f + dx_12 / (dx_01 - dx_12);

    feature.location_ += x * unit_grad;
}

cv::Mat MinFeaturesExtractor::DisplayFeatures(
    const cv::Mat& image, 
    const std::vector<MinFeature2D>& features, 
    cv::Vec3b color) 
{
    cv::Mat result = ConvertToColor(image);
    
    for(auto f : features) {
        cv::Point2f center(f.location_.x(), f.location_.y());
        cv::circle(result, center, 2, color, 1);
        //result.at<cv::Vec3b>(f.location_.y(), f.location_.x()) = color; 
    }

    return result;
}