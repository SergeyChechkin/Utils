// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "Rotation.h"
#include <ceres/jet.h>
#include <ceres/rotation.h>

/// @brief 3D rigid body transformation 
/// @tparam T - scalar type
template<typename T>
class Transformation {
public:
    /// @brief rigid body transformation
    /// @param pose - pose 
    /// @param src - source 3D point 
    /// @param dst - result 3D point
    static void f(const T pose[6], const T src[3], T dst[3]) {
        Rotation<T>::f(pose, src, dst);
        dst[0] += pose[3];
        dst[1] += pose[4];
        dst[2] += pose[5];
    }

    static Eigen::Vector3<T> f(const Eigen::Vector<T, 6>& pose, const Eigen::Vector3<T>& pnt) {
        Eigen::Vector3<T> result;

        Rotation<T>::f(pose.data(), pnt.data(), result.data());
        result[0] += pose[3];
        result[1] += pose[4];
        result[2] += pose[5];

        return result;
    }

    /// @brief pose and point derivativs  
    /// @param pose - pose 
    /// @param point - point
    /// @return - transformd point and derivatives
    static Eigen::Vector3<ceres::Jet<T, 9>> df(const T pose[6], const T src[3]) {
        static const T one = T(1.0);
        static const T zero = T(0.0);
        Eigen::Vector3<ceres::Jet<T, 9>> result;
        auto rot_res = Rotation<T>::df(pose, src);
        result[0].a = rot_res.Rpt[0] + pose[3];
        result[1].a = rot_res.Rpt[1] + pose[4];
        result[2].a = rot_res.Rpt[2] + pose[5];

        result[0].v[0] = rot_res.df_daa[0];
        result[1].v[0] = rot_res.df_daa[1];
        result[2].v[0] = rot_res.df_daa[2];

        result[0].v[1] = rot_res.df_daa[3];
        result[1].v[1] = rot_res.df_daa[4];
        result[2].v[1] = rot_res.df_daa[5];

        result[0].v[2] = rot_res.df_daa[6];
        result[1].v[2] = rot_res.df_daa[7];
        result[2].v[2] = rot_res.df_daa[8];

        result[0].v[3] = one;
        result[1].v[3] = zero;
        result[2].v[3] = zero;

        result[0].v[4] = zero;
        result[1].v[4] = one;
        result[2].v[4] = zero;

        result[0].v[5] = zero;
        result[1].v[5] = zero;
        result[2].v[5] = one;

        result[0].v[6] = rot_res.df_dp[0];
        result[1].v[6] = rot_res.df_dp[1];
        result[2].v[6] = rot_res.df_dp[2];

        result[0].v[7] = rot_res.df_dp[3];
        result[1].v[7] = rot_res.df_dp[4];
        result[2].v[7] = rot_res.df_dp[5];

        result[0].v[8] = rot_res.df_dp[6];
        result[1].v[8] = rot_res.df_dp[7];
        result[2].v[8] = rot_res.df_dp[8];

        return result;
    }

    /// @brief pose derivativs  
    /// @param pose - pose 
    /// @param point - point
    /// @return - transformd point and derivative
    static Eigen::Vector3<ceres::Jet<T, 6>> df_dps(const T pose[6], const T point[3]) {
        static const T one = T(1.0);
        static const T zero = T(0.0);
        Eigen::Vector3<ceres::Jet<T, 6>> result;
        auto rot_res = Rotation<T>::df(pose, point);
        result[0].a = rot_res.Rpt[0] + pose[3];
        result[1].a = rot_res.Rpt[1] + pose[4];
        result[2].a = rot_res.Rpt[2] + pose[5];

        // todo: improve
        result[0].v[0] = rot_res.df_daa[0];
        result[1].v[0] = rot_res.df_daa[1];
        result[2].v[0] = rot_res.df_daa[2];

        result[0].v[1] = rot_res.df_daa[3];
        result[1].v[1] = rot_res.df_daa[4];
        result[2].v[1] = rot_res.df_daa[5];

        result[0].v[2] = rot_res.df_daa[6];
        result[1].v[2] = rot_res.df_daa[7];
        result[2].v[2] = rot_res.df_daa[8];

        result[0].v[3] = one;
        result[1].v[3] = zero;
        result[2].v[3] = zero;

        result[0].v[4] = zero;
        result[1].v[4] = one;
        result[2].v[4] = zero;

        result[0].v[5] = zero;
        result[1].v[5] = zero;
        result[2].v[5] = one;

        return result;
    }


    /// @brief Zero pose derivativs  
    /// @param point - point
    /// @return - transformd point and derivative
    inline static Eigen::Matrix<T, 3, 6> df_dps_zero(const T point[3]) {
        static const T one = T(1.0);
        Eigen::Matrix<T, 3, 6> result;
        result.setZero();

        result(0,1) = point[2];
        result(0,2) = -point[1];
        result(0,3) = one;

        result(1,0) = -point[2];
        result(1,2) = point[0];
        result(1,4) = one;

        result(2,0) = point[1];
        result(2,1) = -point[0];
        result(2,5) = one;

        return result;
    }

    // Ceres Jet versions for Unit test
    static Eigen::Vector3<ceres::Jet<T, 9>> df_cj(const T pose[6], const T point[3]) 
    {
        using JetT = ceres::Jet<double, 9>;
        Eigen::Vector<JetT, 6> pose_j;
        for(int i = 0; i < 6; ++i)  
            pose_j[i] = JetT(pose[i], i);
        
        Eigen::Vector3<JetT> point_j;
        for(int i = 0; i < 3; ++i)  
            point_j[i] = JetT(point[i], 6 + i);

        Eigen::Vector3<JetT> res_j;
        Transformation<JetT>::f(pose_j.data(), point_j.data(), res_j.data());

        return res_j;
    }

    /// @brief Partial derivative by pose only 
    /// @param pose - pose
    /// @param point - point
    /// @return - transformed point with pose partial derivative 
    static Eigen::Vector3<ceres::Jet<T, 6>> df_dps_cj(const T pose[6], const T point[3]) {

        using JetT = ceres::Jet<double, 9>;
        Eigen::Vector3<JetT> res_j = Transformation<JetT>::df_cj(pose, point);

        Eigen::Vector3<ceres::Jet<T, 6>> result;

        for(int i = 0; i < 3; ++i) {
            result[i].a = res_j[i].a;
            for(int j = 0; j < 6; ++j) {
                result[i].v[j] = res_j[i].v[j];
            }
        }

        return result;
    }

    /// @brief Partial derivative by point only 
    /// @param pose - pose
    /// @param point - point 
    /// @return - transformed point with src point partial derivative 
    static Eigen::Vector3<ceres::Jet<T, 3>> df_dpt_cj(const T pose[6], const T point[3]) 
    {
        using JetT = ceres::Jet<double, 9>;
        Eigen::Vector3<JetT> res_j = Transformation<JetT>::df_cj(pose, point);

        Eigen::Vector3<ceres::Jet<T, 3>> result;

        for(int i = 0; i < 3; ++i) {
            result[i].a = res_j[i].a;
            for(int j = 0; j < 3; ++j) {
                result[i].v[j] = res_j[i].v[j+6];
            }
        }

        return result;
    }

    static Eigen::Transform<T, 3, Eigen::Isometry> Convert(const T pose[6]) {
        Eigen::Matrix<T, 3, 3> rot_mat;
        ceres::AngleAxisToRotationMatrix<T>(pose, rot_mat.data());

        Eigen::Transform<T, 3, Eigen::Isometry> result;
        result.linear() = rot_mat;
        result.translation() << pose[3], pose[4], pose[5];

        return result;
    }

    static Eigen::Vector<T, 6> Convert(const Eigen::Transform<T, 3, Eigen::Isometry>& ism_pose) {
        Eigen::Vector<T, 6> result;
        const Eigen::Matrix<T, 3, 3> rot_mat = ism_pose.linear();
        const Eigen::Vector3<T> t = ism_pose.translation();

        ceres::RotationMatrixToAngleAxis<T>(rot_mat.data(), result.data());
        result[3] = t[0]; 
        result[4] = t[1]; 
        result[5] = t[2]; 

        return result;
    }
};
