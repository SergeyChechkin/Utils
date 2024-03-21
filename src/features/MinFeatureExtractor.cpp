#include "utils/features/MinFeature/MinFeatureExtractor.h"
#include <spatial_hash/SpatialHash2DVector.h>
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
    cv::dilate(grad_norm_sqr, dilate_grad_norm_sqr, cv::Mat(), cv::Point(-1,-1), 1);

    const int height = dilate_grad_norm_sqr.rows;
    const int width = dilate_grad_norm_sqr.cols;
    const int offset = image.PatchSize() / 2;
    const int result_size = height * width * config.square_count_ / (config.square_size_ * config.square_count_);
    // In order to have evenly distributed features
    // we limit number of features in each sector        
    std::vector<std::pair<float, MinFeature2D>> all_features;
    all_features.reserve(result_size); 
    SpatialHashTable2DVector<float, size_t> hash_table(config.square_size_); 

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
                hash_table.Add(feature.location_.data(), all_features.size());
                all_features.push_back({dI_norm, feature});
            }            
        }
    }

    // select fixed number of features from each sector
    std::vector<MinFeature2D> result;
    result.reserve(result_size);
    
    for(const auto& cell : hash_table.GetTable()) {
        std::map<float, MinFeature2D, std::greater<float>> cell_features;
        for(auto idx : cell.second) {
            cell_features.insert(all_features[idx]);
        }

        int count = config.square_count_;
        for(const auto& itr : cell_features) {
            result.push_back(itr.second);
            
            --count;
            if (!count)
                break;
        }
    }

    return result;
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