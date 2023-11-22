#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

template<typename T>
inline Eigen::Vector3<T> TransformPoint(const T pose[6], const T src[3]) {
    Eigen::Vector3<T> result;
    ceres::AngleAxisRotatePoint(pose, src, result.data());
    return result + Eigen::Vector3<T>(pose + 3);
}

bool Optimize(
        ceres::Problem& problem,
        bool verbal = true,
        int max_iterations = 100,
        double threshold = 1e-6);

Eigen::MatrixX<double> Convert(const ceres::CRSMatrix& src);
