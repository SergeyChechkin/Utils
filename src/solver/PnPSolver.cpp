/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#include "utils/solver/PnPSolver.h"
#include "utils/solver/Transformation.h"
#include "utils/solver/PerspectiveProjection.h"

#include <Eigen/Eigenvalues>
#include <ceres/jet.h>

void PnPSolver::SolvePose(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector2d>& features, 
    Eigen::Vector<double, 6>& pose, 
    const Cofiguration& config) 
{
    const size_t factor_count = points.size();
    double last_mean_cost = 0;
    
    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;   

    Eigen::Matrix<double, 6, 6> factor_H;
    Eigen::Matrix<double, 6, 1> factor_b;
    double factor_cost;  
    
    for(size_t itr = 0; itr < config.max_iterations; ++itr) {
        H = Eigen::Matrix<double, 6, 6>::Zero();
        b = Eigen::Matrix<double, 6, 1>::Zero();   
        double cost = 0;

        for(size_t i = 0; i < factor_count; ++i) {
            GetPoseFactor(points[i], features[i], pose, factor_H, factor_b, factor_cost);

            H += factor_H;
            b += factor_b;
            cost += factor_cost; 
        }

        auto dx = H.ldlt().solve(b);
        pose += dx;
        
        double mean_cost = cost / factor_count;
        double cost_diff = last_mean_cost - mean_cost;
        double cost_change = std::abs(cost_diff) / mean_cost;

        if (config.verbal) {
            std::cout << "Iteration " << itr << ": "; 
            std::cout << "mean cost - " << cost / factor_count << ", ";
            std::cout << "step - " << dx.norm() << std::endl;
        }

        if (mean_cost < config.min_cost)
            break;
        
        if (cost_change < config.min_cost_change)
            break;
        
        last_mean_cost = mean_cost;
    }
}

void PnPSolver::GetPoseFactor(
    const Eigen::Vector3d& point,
    const Eigen::Vector2d& feature, 
    const Eigen::Vector<double, 6>& pose,
    Eigen::Matrix<double, 6, 6>& H,
    Eigen::Matrix<double, 6, 1>& b,   
    double& cost) 
{
    auto prj_J = PerspectiveProjection<double>::df_dps(pose.data(), point.data());

    Eigen::Matrix<double, 2, 6> J;
    J.row(0) = prj_J[0].v;
    J.row(1) = prj_J[1].v;

    const Eigen::Vector2d err(prj_J[0].a - feature[0], prj_J[1].a - feature[1]);

    H = J.transpose() * J;
    b = -err.transpose() * J;
    cost = err.squaredNorm(); 
}

void PnPSolver::GetPoseEpipolarFactor(
    const Eigen::Vector3d& point,
    double depth_info,
    double tan_info,
    const Eigen::Vector2d& feature, 
    const Eigen::Vector<double, 6>& pose,
    Eigen::Matrix<double, 6, 6>& H,
    Eigen::Matrix<double, 6, 1>& b,   
    double& cost) 
{
    using JetT = ceres::Jet<double, 6>;

    JetT pose_J[6];
    for(int i = 0; i < 6; ++i)
        pose_J[i] = JetT(pose[i], i);

    const Eigen::Vector2<JetT> plane_point = PerspectiveProjection<double>::df_dps(pose.data(), point.data());
    const Eigen::Vector2<JetT> error(plane_point[0] - feature[0], plane_point[1] - feature[1]);

    const Eigen::Vector3<JetT> epipole(pose_J + 3);
    Eigen::Vector2<JetT> plane_epipole(JetT(0), JetT(0));
    if (abs(epipole[2]) > std::numeric_limits<JetT>::epsilon()) {
        plane_epipole[0] = epipole[0] / epipole[2];
        plane_epipole[1] = epipole[1] / epipole[2];
    } 

    const Eigen::Vector2<JetT> epipolar_vector = (plane_point - plane_epipole).normalized();
    const Eigen::Vector2<JetT> epipolar_vector_tan(epipolar_vector[1], -epipolar_vector[0]);

    JetT residuals[2];
    residuals[0] = JetT(depth_info) * epipolar_vector.dot(error);
    residuals[1] = JetT(tan_info) * epipolar_vector_tan.dot(error);

    Eigen::Matrix<double, 2, 6> J;
    J.row(0) = residuals[0].v;
    J.row(1) = residuals[1].v;

    const Eigen::Vector2d err(residuals[0].a, residuals[1].a);

    H = J.transpose() * J;
    b = -err.transpose() * J;
    cost = err.squaredNorm(); 
}