/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <opencv2/calib3d.hpp>
#include <Eigen/Geometry>

class HomographySolver {
public:
    struct Cofiguration {
        double focal_ = 1.0;
        cv::Point2d pp_ = cv::Point2d(0, 0);
        int method_ = cv::RANSAC;
        double prob_ = 0.999;
        double threshold_ = 0.001;   // unit plane units
        int max_iters_ = 1000;
        size_t depth_test_itr_ = 100; 

        const double min_cost = 1.0E-6;
        const double min_cost_change = 1.0E-8;
        const size_t max_iterations = 50;
    };
public:
    static bool Solve(
        const std::vector<Eigen::Vector2d>& prev, 
        const std::vector<Eigen::Vector2d>& next, 
        Eigen::Isometry3d& pose, 
        const Cofiguration& config);

    static bool Solve(
        const std::vector<Eigen::Vector2d>& prev, 
        const std::vector<Eigen::Vector2d>& next, 
        Eigen::Isometry3d& pose,
        std::vector<double>& depths, 
        const Cofiguration& config);

    static bool Solve_Ceres(
        const std::vector<Eigen::Vector2d>& prev, 
        const std::vector<Eigen::Vector2d>& next, 
        Eigen::Vector3d& aa,  
        Eigen::Vector3d& t);

    static bool Solve_Ceres_qt(
        const std::vector<Eigen::Vector2d>& prev, 
        const std::vector<Eigen::Vector2d>& next, 
        Eigen::Vector4d& q,  
        Eigen::Vector3d& t);
private:
    static size_t ComputeEssentialMatrix(
        const std::vector<Eigen::Vector2d>& prev, 
        const std::vector<Eigen::Vector2d>& next, 
        Eigen::Matrix3d& E, 
        const Cofiguration& config);

    static void DecomposeEssentialMatrix(
        const Eigen::Matrix3d& E,
        Eigen::Matrix3d& R1,
        Eigen::Matrix3d& R2,
        Eigen::Vector3d& t);
};