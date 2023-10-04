#include "utils/features/KLT.h"
#include "utils/image/BilinearInterpolation.h"
#include <glog/logging.h>

KLT::KLT(
    const ImageWithGradient& src_img, 
    const ImageWithGradient& dst_img, 
    size_t patch_size, 
    size_t max_itr) 
: src_img_(src_img)
, dst_img_(dst_img)
, patch_size_(patch_size)
, max_itr_(max_itr)
, center_offset_(patch_size_ / 2)
, err_size_(patch_size_ * patch_size_)
{
    CHECK_EQ(src_img_.gray_.size(), dst_img_.gray_.size());
    CHECK_EQ(src_img_.gray_.type(), image_type_);   // grayscale only
    CHECK_EQ(dst_img_.gray_.type(), image_type_);   // grayscale only

    top_left_.x = border_;
    top_left_.y = border_;
    btm_rght_.x = src_img_.gray_.cols - border_;
    btm_rght_.y = src_img_.gray_.rows - border_;
} 

bool KLT::CheckPointLocation(const cv::Point2f& pnt) const
{
    return !(pnt.x < top_left_.x || pnt.y < top_left_.y || pnt.x >= btm_rght_.x || pnt.y >= btm_rght_.y);
}

void KLT::TrackPoint(
    const cv::Point2f& src_pnt, 
    cv::Point2f& dst_pnt) const
{
    float prev_error = std::numeric_limits<float>::max();

    const cv::Size cv_patch_size(patch_size_, patch_size_);  

    for(int itr = 0; itr < max_itr_; ++itr) {
        
        Eigen::Matrix<double, 2, 2> H = Eigen::Matrix<double, 2, 2>::Zero();
        Eigen::Matrix<double, 2, 1> b = Eigen::Matrix<double, 2, 1>::Zero();
        float error = 0;

        for(int v_patch = 0; v_patch < patch_size_; ++v_patch)
        {
            for(int u_patch = 0; u_patch < patch_size_; ++u_patch)
            {
                const cv::Point2f offset(v_patch - center_offset_, u_patch - center_offset_);
                const cv::Point2f src_patch_pnt = src_pnt + offset;
                const cv::Point2f dst_patch_pnt = dst_pnt + offset;

                if (!CheckPointLocation(src_patch_pnt) || !CheckPointLocation(dst_patch_pnt))
                    continue;

                /// TODO: src point is constant, precompute outside 
                const float src_I = BilinearValue_<float, PixelT>(src_img_.gray_, &(src_patch_pnt.x));
                const float src_dx = BilinearValue_<float, float>(src_img_.grad_x_, &(src_patch_pnt.x));
                const float src_dy = BilinearValue_<float, float>(src_img_.grad_y_, &(src_patch_pnt.x));

                const float dst_I = BilinearValue_<float, PixelT>(dst_img_.gray_, &(dst_patch_pnt.x));
                const float dst_dx = BilinearValue_<float, float>(dst_img_.grad_x_, &(dst_patch_pnt.x));
                const float dst_dy = BilinearValue_<float, float>(dst_img_.grad_y_, &(dst_patch_pnt.x));

                const float s_x = src_dx + dst_dx; 
                const float s_y = src_dy + dst_dy; 

                const Eigen::Matrix<double, 2, 1> J = {s_x, s_y};
                const double res = 2.0 * (dst_I - src_I); 

                H += J * J.transpose();
                b -= res * J;
                
                error += res * res;
            }
        }

        const auto dw = H.ldlt().solve(b);
        dst_pnt.x += dw[0];
        dst_pnt.y += dw[1];
        error /= err_size_;

        const float error_change = prev_error - error;
        if (error_change < min_error_change_ || error < min_error_)
            break;

        prev_error = error;
    }
}

void KLT::Track(
    const std::vector<cv::Point2f>& src_pnts, 
    std::vector<cv::Point2f>& dst_pnts) const
{
    CHECK_EQ(src_pnts.size(), dst_pnts.size());
    const size_t pnts_size = src_pnts.size();
    
    /// TODO: parallel computation
    for(size_t pnt_idx = 0; pnt_idx < pnts_size; ++pnt_idx)
    {
        const auto src_pnt = src_pnts[pnt_idx];
        const auto dst_pnt = dst_pnts[pnt_idx];
        if (CheckPointLocation(src_pnt) && CheckPointLocation(dst_pnt)) {
            TrackPoint(src_pnts[pnt_idx], dst_pnts[pnt_idx]);
        }
    }
}

double KLT::ComputeGain(
    const std::vector<cv::Point2f>& src_pnts, 
    const std::vector<cv::Point2f>& dst_pnts) const
{
    CHECK_EQ(src_pnts.size(), dst_pnts.size());
    const size_t pnts_size = src_pnts.size();
    const size_t mat_size = 2 * pnts_size;
    const cv::Size cv_patch_size(patch_size_, patch_size_);            
    
    Eigen::MatrixXd W(mat_size, 1);
    Eigen::MatrixXd V(mat_size, 1);
    Eigen::MatrixXd U_inv(mat_size, mat_size);
    W.setZero();
    V.setZero();
    U_inv.setZero();
    
    double lambda = 0;
    double m = 0;

    /// TODO: parallel computation
    for(size_t pnt_idx = 0; pnt_idx < pnts_size; ++pnt_idx) {
        const auto mat_idx_0 = 2 * pnt_idx; 
        const auto mat_idx_1 = 2 * pnt_idx + 1; 
        
        Eigen::Matrix2d U_pnt;
        U_pnt.setZero();

        const auto src_pnt = src_pnts[pnt_idx];
        const auto dst_pnt = dst_pnts[pnt_idx];
        
        if (!CheckPointLocation(src_pnt) || !CheckPointLocation(dst_pnt))
            continue;

        int count = 0;

        for(int v_patch = 0; v_patch < patch_size_; ++v_patch)
        {
            for(int u_patch = 0; u_patch < patch_size_; ++u_patch)
            {
                const cv::Point2f offset(v_patch - center_offset_, u_patch - center_offset_);
                const cv::Point2f src_patch_pnt = src_pnt + offset;
                const cv::Point2f dst_patch_pnt = dst_pnt + offset;

                if (!CheckPointLocation(src_patch_pnt) || !CheckPointLocation(dst_patch_pnt))
                    continue;

                ++count;

                /// TODO: src point is constant, precompute outside 
                const float src_I = std::max<float>(1.0f, BilinearValue_<float, PixelT>(src_img_.gray_, &(src_patch_pnt.x)));
                const float inv_src_I = 1.0 / src_I;
                const float src_dx = BilinearValue_<float, float>(src_img_.grad_x_, &(src_patch_pnt.x));
                const float src_dy = BilinearValue_<float, float>(src_img_.grad_y_, &(src_patch_pnt.x));

                const float dst_I = std::max<float>(1.0f, BilinearValue_<float, PixelT>(dst_img_.gray_, &(dst_patch_pnt.x)));
                const float inv_dst_I = 1.0 / dst_I;
                const float dst_dx = BilinearValue_<float, float>(dst_img_.grad_x_, &(dst_patch_pnt.x));
                const float dst_dy = BilinearValue_<float, float>(dst_img_.grad_y_, &(dst_patch_pnt.x));

                const float a = src_dx * inv_src_I + dst_dx * inv_dst_I; 
                const float b = src_dy * inv_src_I + dst_dy * inv_dst_I; 
                const float beta = std::log(dst_I) - std::log(src_I);

                const Eigen::Matrix<double, 2, 1> J = {a, b};
                U_pnt += 0.5 * J * J.transpose();
                W(mat_idx_0, 0) -= a;
                W(mat_idx_1, 0) -= b;
                V(mat_idx_0, 0) -= beta * a;
                V(mat_idx_1, 0) -= beta * b;
                
                lambda += 2;
                m += 2 * beta;
            }
        }

        if (count > 0) {
            U_inv.block<2,2>(mat_idx_0, mat_idx_0) = U_pnt.inverse();  
        }
    } 

    // compute exposure difference 
    /// TODO: use sparse matrix multiplicationm? U_inv is sparse. 
    const auto Wt_U_inv = -W.transpose() * U_inv;
    const auto Wt_U_inv_W = Wt_U_inv * W;
    const auto Wt_U_inv_V = Wt_U_inv * V;
    const double exposure_dif_log = (Wt_U_inv_V(0, 0) + m) / (Wt_U_inv_W(0, 0) + lambda);

    return std::exp(exposure_dif_log);
}

void KLT::TrackGainInvariant(
    const std::vector<cv::Point2f>& src_pnts, 
    std::vector<cv::Point2f>& dst_pnts, 
    double& gain) const
{
    CHECK_EQ(src_pnts.size(), dst_pnts.size());

    const cv::Size cv_patch_size(patch_size_, patch_size_);            
    const size_t pnts_size = src_pnts.size();
    const size_t mat_size = 2 * pnts_size;
    
    for(int itr = 0; itr < max_itr_; ++itr) 
    {
        Eigen::MatrixXd W(mat_size, 1);
        Eigen::MatrixXd V(mat_size, 1);
        Eigen::MatrixXd U_inv(mat_size, mat_size);
        W.setZero();
        V.setZero();
        U_inv.setZero();
        
        double lambda = 0;
        double m = 0;

        /// TODO: parallel computation
        for(size_t pnt_idx = 0; pnt_idx < pnts_size; ++pnt_idx) {
            const auto mat_idx_0 = 2 * pnt_idx; 
            const auto mat_idx_1 = 2 * pnt_idx + 1; 
            
            Eigen::Matrix2d U_pnt;
            U_pnt.setZero();

            const auto src_pnt = src_pnts[pnt_idx];
            const auto dst_pnt = dst_pnts[pnt_idx];
            
            if (!CheckPointLocation(src_pnt) || !CheckPointLocation(dst_pnt))
                    continue;

            int count = 0;

            for(int v_patch = 0; v_patch < patch_size_; ++v_patch)
            {
                for(int u_patch = 0; u_patch < patch_size_; ++u_patch)
                {
                    const cv::Point2f offset(v_patch - center_offset_, u_patch - center_offset_);
                    const cv::Point2f src_patch_pnt = src_pnt + offset;
                    const cv::Point2f dst_patch_pnt = dst_pnt + offset;

                    if (!CheckPointLocation(src_patch_pnt) || !CheckPointLocation(dst_patch_pnt))
                        continue;

                    ++count;

                    /// TODO: src point is constant, precompute outside 
                    const float src_I = std::max<float>(1.0f, BilinearValue_<float, PixelT>(src_img_.gray_, &(src_patch_pnt.x)));
                    const float inv_src_I = 1.0 / src_I;
                    const float src_dx = BilinearValue_<float, float>(src_img_.grad_x_, &(src_patch_pnt.x));
                    const float src_dy = BilinearValue_<float, float>(src_img_.grad_y_, &(src_patch_pnt.x));

                    const float dst_I = std::max<float>(1.0f, BilinearValue_<float, PixelT>(dst_img_.gray_, &(dst_patch_pnt.x)));
                    const float inv_dst_I = 1.0 / dst_I;
                    const float dst_dx = BilinearValue_<float, float>(dst_img_.grad_x_, &(dst_patch_pnt.x));
                    const float dst_dy = BilinearValue_<float, float>(dst_img_.grad_y_, &(dst_patch_pnt.x));

                    const float a = src_dx * inv_src_I + dst_dx * inv_dst_I; 
                    const float b = src_dy * inv_src_I + dst_dy * inv_dst_I; 
                    const float beta = std::log(dst_I) - std::log(src_I);

                    const Eigen::Matrix<double, 2, 1> J = {a, b};
                    U_pnt += 0.5 * J * J.transpose();
                    W(mat_idx_0, 0) -= a;
                    W(mat_idx_1, 0) -= b;
                    V(mat_idx_0, 0) -= beta * a;
                    V(mat_idx_1, 0) -= beta * b;
                    
                    lambda += 2;
                    m += 2 * beta;
                }
            }
            
            if (count > 0) {
                U_inv.block<2,2>(mat_idx_0, mat_idx_0) = U_pnt.inverse();  
            }
        } 

        // compute exposure difference 
        /// TODO: use sparse matrix multiplicationm. U_inv is sparse 
        const auto Wt_U_inv = -W.transpose() * U_inv;
        const auto Wt_U_inv_W = Wt_U_inv * W;
        const auto Wt_U_inv_V = Wt_U_inv * V;
        const double exposure_dif_log = (Wt_U_inv_V(0, 0) + m) / (Wt_U_inv_W(0, 0) + lambda);

        // compute displacements
        /// TODO: parallel computation
        for(size_t pnt_idx = 0; pnt_idx < pnts_size; ++pnt_idx) {
            const auto mat_idx_0 = 2 * pnt_idx; 
            
            const auto U_pnt_inv = U_inv.block<2,2>(mat_idx_0, mat_idx_0);
            const auto V_pnt = V.block<2, 1>(mat_idx_0, 0);
            const auto W_pnt = W.block<2, 1>(mat_idx_0, 0);

            const auto displacement = U_pnt_inv * (V_pnt - exposure_dif_log * W_pnt); 

            dst_pnts[pnt_idx].x += displacement[0];
            dst_pnts[pnt_idx].y += displacement[1];
        }

        gain = std::exp(exposure_dif_log);
    }
}
