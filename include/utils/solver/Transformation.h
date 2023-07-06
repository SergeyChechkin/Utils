// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "Rotation.h"
#include <ceres/jet.h>
#include <ceres/rotation.h>

template<typename T>
class Transformation {
public:
    static void f(const T pose[6], const T src[3], T dst[3]) {
        Rotation<T>::f(pose, src, dst);
        dst[0] += pose[3];
        dst[1] += pose[4];
        dst[2] += pose[5];
    }

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

    static Eigen::Vector3<ceres::Jet<T, 6>> df_dps_cj(const T pose[6], const T point[3]) {

        using JetT = ceres::Jet<double, 9>;
        Eigen::Vector<JetT, 6> pose_j;
        Eigen::Vector3<JetT> pt_j;
        for(int i = 0; i < 6; ++i)  {
            pose_j[i] = JetT(pose[i], i);
        }

        for(int i = 0; i < 3; ++i)  {
            pt_j[i] = JetT(point[i]);
        }

        Eigen::Vector3<JetT> res_j;
        Transformation<JetT>::f(pose_j.data(), pt_j.data(), res_j.data());

        Eigen::Vector3<ceres::Jet<T, 6>> result;

        for(int i = 0; i < 3; ++i) {
            result[i].a = res_j[i].a;
            for(int j = 0; j < 6; ++j) {
                result[i].v[j] = res_j[i].v[j];
            }
        }

        return result;
    }

    static Eigen::Vector3<ceres::Jet<T, 9>> df_dps_dpt_cj(const T pose[6], const T point[3]) 
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

    static Eigen::Vector3<ceres::Jet<T, 3>> df_dpt_cj(const T pose[6], const T point[3]) 
    {
        using JetT = ceres::Jet<double, 9>;
        Eigen::Vector<JetT, 6> pose_j;
        Eigen::Vector3<JetT> pt_j;
        for(int i = 0; i < 6; ++i)  {
            pose_j[i] = JetT(pose[i]);
        }

        for(int i = 0; i < 3; ++i)  {
            pt_j[i] = JetT(point[i], i + 6);
        }

        Eigen::Vector3<JetT> res_j;
        Transformation<JetT>::f(pose_j.data(), pt_j.data(), res_j.data());

        Eigen::Vector3<ceres::Jet<T, 3>> result;

        for(int i = 0; i < 3; ++i) {
            result[i].a = res_j[i].a;
            for(int j = 0; j < 3; ++j) {
                result[i].v[j] = res_j[i].v[j+6];
            }
        }

        return result;
    }

};
