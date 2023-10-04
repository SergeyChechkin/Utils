#include "utils/features/InverseLK_Pose.h"

#include <utils/image/BilinearInterpolation.h>
#include <utils/solver/Transformation.h>
#include <utils/solver/PerspectiveProjection.h>

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <iostream>

InverseLK_Pose::InverseLK_Pose(const CameraModelT& cm, const ImageWithGradient& src_img, const ImageWithGradient& dst_img) 
: cm_(cm)
, src_img_(src_img)
, dst_img_(dst_img)
{
    CHECK_EQ(src_img_.gray_.size(), dst_img_.gray_.size());
    CHECK_EQ(src_img_.gray_.type(), image_type_);   // grayscale only
    CHECK_EQ(dst_img_.gray_.type(), image_type_);   // grayscale only

    top_left_ << border_, border_;
    btm_rght_ << src_img_.gray_.cols - border_, src_img_.gray_.rows - border_;

    // TODO: add focal lenght access to camera interface 
    const auto& cm_param = cm_.Params();
    f_ = 0.5 * (cm_param[0] + cm_param[1]);
} 

Eigen::Vector2d InverseLK_Pose::PatchPointFromIndex(int idx) {
    return {idx % patch_size_ - center_offset_, idx / patch_size_ - center_offset_}; 
}

bool InverseLK_Pose::CheckPointLocation(const Eigen::Vector2d& pnt) const
{
    return !(pnt[0] < top_left_[0] || pnt[1] < top_left_[1] || pnt[0] >= btm_rght_[0] || pnt[1] >= btm_rght_[1]);
}

bool InverseLK_Pose::Track(
    const std::vector<Eigen::Vector2d>& src_img,
    const std::vector<Eigen::Vector3d>& src_obj, 
    Eigen::Isometry3d& pose) const 
{
    CHECK_EQ(src_img.size(), src_obj.size());

    const size_t pnts_size = src_img.size();
    const double inv_pnts_size = 1.0 / double(pnts_size);
    

    const int mat_size = err_size_ * pnts_size;
    Eigen::MatrixXd Jt(6, mat_size);    
    Eigen::VectorXd src_I(mat_size);
    
    Jt.setZero();
    src_I.setZero();

    // TODO: parallel computation
    for(size_t pnt_idx = 0; pnt_idx < pnts_size; ++pnt_idx) {
        const Eigen::Vector2d src_img_point = src_img[pnt_idx];
        const Eigen::Vector3d w_point = src_obj[pnt_idx];
        
        const Eigen::Matrix<double, 2, 6> J_dps_zero = PerspectiveProjection<double>::df_dps_zero(f_, w_point); 

        for(int i = 0; i < err_size_; ++i) {
            const Eigen::Vector2d src_patch_point = src_img_point + PatchPointFromIndex(i);
        
            if (!CheckPointLocation(src_patch_point))
                continue;

            const double I = BilinearValue_<double, PixelT>(src_img_.gray_, src_patch_point.data()); 
            const double dx = BilinearValue_<double, float>(src_img_.grad_x_, src_patch_point.data());
            const double dy = BilinearValue_<double, float>(src_img_.grad_y_, src_patch_point.data());
            const Eigen::Matrix<double, 1, 2> dI(dx, dy);

            const int row_idx = pnt_idx * err_size_ + i; 
            
            src_I[row_idx] = I;
            Jt.block<6, 1>(0, row_idx) = dI * J_dps_zero;
        }
    }

    const Eigen::Matrix<double, 6, 6> H = Jt * Jt.transpose();
    Eigen::VectorXd errs(err_size_ * pnts_size);

    float prev_error = std::numeric_limits<float>::max();

    size_t itr = 0;

    for(; itr < max_itr_; ++itr) {
        errs.setZero();

        // TODO: parallel computation
        for(size_t pnt_idx = 0; pnt_idx < pnts_size; ++pnt_idx) {
            const Eigen::Vector2d src_img_point = src_img[pnt_idx];
            const Eigen::Vector3d src_w_point = src_obj[pnt_idx];
            const Eigen::Vector3d dst_w_point = pose * src_w_point;
            const Eigen::Vector2d dst_img_point = cm_.Project(dst_w_point);

            for(size_t i = 0; i < err_size_; ++i) {
                const Eigen::Vector2d src_patch_point = src_img_point + PatchPointFromIndex(i);
                const Eigen::Vector2d dst_patch_point = dst_img_point + PatchPointFromIndex(i);
                
                if (!CheckPointLocation(src_patch_point) || !CheckPointLocation(dst_patch_point))
                    continue;

                const double I = BilinearValue_<double, PixelT>(dst_img_.gray_, dst_patch_point.data()); 
                
                const int row_idx = pnt_idx * err_size_ + i; 
                errs[row_idx] = src_I[row_idx] - I;
            }       
        }    

        double error = inv_pnts_size * errs.transpose() * errs;
        const float error_change = std::abs(prev_error - error);

        std::cout << itr << " error " << error << " " << error_change << std::endl;
    
        if (error_change < min_error_change_ || error < min_error_)
            break;

        const Eigen::Matrix<double, 6, 1> b = Jt * errs;
        const Eigen::Vector<double, 6> dw = -H.ldlt().solve(b);
        pose = pose * (Transformation<double>::Convert(dw.data()).inverse());

        std::cout << "dw: " << dw.transpose() << std::endl;
        std::cout << std::endl;

        prev_error = error;
    }

    return itr < max_itr_;
}