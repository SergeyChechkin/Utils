// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "Rotation.h"
#include <ceres/jet.h>

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
        result[0].a = rot_res.Rpt[0] += pose[3];
        result[1].a = rot_res.Rpt[1] += pose[4];
        result[2].a = rot_res.Rpt[2] += pose[5];

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
    static Eigen::Vector3<ceres::Jet<T, 6>> df_dp(const T pose[6], const T point[3]) {
        static const T one = T(1.0);
        static const T zero = T(0.0);
        Eigen::Vector3<ceres::Jet<T, 6>> result;
        auto rot_res = Rotation<T>::df(pose, point);
        result[0].a = rot_res.Rpt[0] += pose[3];
        result[1].a = rot_res.Rpt[1] += pose[4];
        result[2].a = rot_res.Rpt[2] += pose[5];

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
};
