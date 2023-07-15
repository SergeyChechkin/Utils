/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#include "utils/solver/HomographySolver.h"
#include "utils/geometry/Triangulation.h"
#include <opencv2/core/eigen.hpp>

#include <glog/logging.h>

bool HomographySolver::Solve(
    const std::vector<Eigen::Vector2d>& prev, 
    const std::vector<Eigen::Vector2d>& next, 
    Eigen::Isometry3d& pose, 
    const Cofiguration& config) 
{
    CHECK_GE(prev.size(), 5);
    CHECK_EQ(prev.size(), prev.size());

    Eigen::Matrix3d E;
    ComputeEssentialMatrix(prev, next, E, config);

    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d t;
    DecomposeEssentialMatrix(E, R1, R2, t);

    Eigen::Matrix3d Rs[4] = {R1, R1, R2, R2};
    Eigen::Vector3d ts[4] = {t, -t, t, -t};
    size_t res_counts = 0;

    for(size_t j = 0; j < 4; ++j) {
        Eigen::Isometry3d T;
        T.linear() = Rs[j];
        T.translation() = ts[j];
        size_t counts = 0;

        size_t size = std::min(config.depth_test_itr_, prev.size());
        for(size_t i = 0; i < size; ++i) {
            double d_prev, d_next;
            if (TriangulatePointDepths(prev[i], T, next[i], d_prev, d_next) && d_prev > 0 && d_next > 0)  {
                ++counts;
            }
        }

        if (res_counts < counts) {
            res_counts = counts;
            pose = T;
        }
    }

    return res_counts >= 5; 
}

bool HomographySolver::Solve(
    const std::vector<Eigen::Vector2d>& prev, 
    const std::vector<Eigen::Vector2d>& next, 
    Eigen::Isometry3d& pose,
    std::vector<double>& depths, 
    const Cofiguration& config)
{
    CHECK_GE(prev.size(), 5);
    CHECK_EQ(prev.size(), prev.size());

    Eigen::Matrix3d E;
    ComputeEssentialMatrix(prev, next, E, config);

    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d t;
    DecomposeEssentialMatrix(E, R1, R2, t);

    Eigen::Matrix3d Rs[4] = {R1, R1, R2, R2};
    Eigen::Vector3d ts[4] = {t, -t, t, -t};
    size_t res_counts = 0;


    for(size_t j = 0; j < 4; ++j) {
        Eigen::Isometry3d T;
        T.linear() = Rs[j];
        T.translation() = ts[j];
        size_t counts = 0;
        std::vector<double> dpts(prev.size(), 1);

        for(size_t i = 0; i < prev.size(); ++i) {
            double d_prev, d_next;
            if (TriangulatePointDepths(prev[i], T, next[i], d_prev, d_next) && d_prev > 0 && d_next > 0)  {
                ++counts;
                dpts[i] = d_prev;
            }
        }

        if (res_counts < counts) {
            res_counts = counts;
            pose = T;
            depths = dpts;
        }
    }

    return res_counts >= 5; 
}

inline std::vector<cv::Point2f> Convert(const std::vector<Eigen::Vector2d>& src) {
    std::vector<cv::Point2f> result(src.size());

    for(size_t i = 0; i < src.size(); ++i) {
       const auto& src_p = src[i];
       auto& dst_p = result[i];
       dst_p.x = src_p[0];
       dst_p.y = src_p[1];
    }

    return result;
}

size_t HomographySolver::ComputeEssentialMatrix(
    const std::vector<Eigen::Vector2d>& prev, 
    const std::vector<Eigen::Vector2d>& next, 
    Eigen::Matrix3d& E, 
    const Cofiguration& config) 
{
    const std::vector<cv::Point2f> prev_cv = Convert(prev); 
    const std::vector<cv::Point2f> next_cv = Convert(next);
    std::vector<uchar> mask; 

    auto E_cv = cv::findEssentialMat(
        prev_cv, next_cv, 
        config.focal_, 
        config.pp_, 
        config.method_, 
        config.prob_, 
        config.threshold_, 
        config.max_iters_,
        mask);
    cv::cv2eigen(E_cv, E);

    return std::accumulate(mask.begin(), mask.end(), 0);
}

void HomographySolver::DecomposeEssentialMatrix(
    const Eigen::Matrix3d& E,
    Eigen::Matrix3d& R1,
    Eigen::Matrix3d& R2,
    Eigen::Vector3d& t) 
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV().transpose();

    //if (U.determinant() < 0) { U *= -1;}
    //if (V.determinant() < 0) { V *= -1;}

    Eigen::Matrix3d W;
    W << 0, 1, 0, -1, 0, 0, 0, 0, 1;

    R1 = (U * W * V).transpose();
    R2 = (U * W.transpose() * V).transpose();
    t = U.col(2).normalized();
}
