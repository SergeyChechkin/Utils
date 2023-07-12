/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Core>

/// @brief Perspective-n-Point transformation solver
class PnPSolver {
public:
    struct Cofiguration {
        const double min_cost = 1.0E-6;
        const double min_cost_change = 1.0E-8;
        const size_t max_iterations = 50;
        bool verbal = true;
    };

    /// @brief Solve frame pose 
    /// @param points - 3D points
    /// @param features - projections
    /// @param pose - frame pose
    /// @param config - configuration
    static void SolvePose(
        const std::vector<Eigen::Vector3d>& points,
        const std::vector<Eigen::Vector2d>& features, 
        Eigen::Vector<double, 6>& pose, 
        const Cofiguration& config);

    /// @brief Perspective-n-Point factor, direct error between 3D point projection and feature 
    /// @param point - 3D point
    /// @param feature - projection 
    /// @param pose - frame pose
    /// @param H - Hessian, Jt x J   
    /// @param b - -et x J 
    /// @param cost - et x e; 
    static void GetPoseFactor(
        const Eigen::Vector3d& point,
        const Eigen::Vector2d& feature, 
        const Eigen::Vector<double, 6>& pose,
        Eigen::Matrix<double, 6, 6>& H,
        Eigen::Matrix<double, 6, 1>& b,   
        double& cost);

    /// @brief Perspective-n-Point factor in case of depth unsertanty.  
    /// Error is weighted along and perpendicular epipolar line.    
    /// @param point - 3D point
    /// @param depth_info - Info (inverse variance) along epipolar line.  
    /// @param tan_info - Info (inverse variance) perpendicular epipolar line. 
    /// @param feature - projection
    /// @param pose - frame pose
    /// @param H - Hessian, Jt x J   
    /// @param b - -et x J 
    /// @param cost - et x e; 
    static void GetPoseEpipolarFactor(
        const Eigen::Vector3d& point,
        double depth_info,
        double tan_info,
        const Eigen::Vector2d& feature, 
        const Eigen::Vector<double, 6>& pose,
        Eigen::Matrix<double, 6, 6>& H,
        Eigen::Matrix<double, 6, 1>& b,   
        double& cost);
};