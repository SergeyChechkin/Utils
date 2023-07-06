/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "Functions.h"
#include "Transformation.h"

#include <Eigen/Core>
#include <ceres/jet.h>
#include <cmath>

/// @brief Unit plane projection
/// @tparam T - scalar type
template<typename T>
class PerspectiveProjection{ 
public:
    /// @brief unit plane projection 
    /// @param pose - pose 
    /// @param pnt - 3D point 
    /// @param prj - 2D projection
    static void f(const T pose[6], const T pnt[3], T prj[2]) {
        T pnt_trans[3];
        Transformation<T>::f(pose, pnt, pnt_trans);

        prj[0] = pnt_trans[0] / pnt_trans[2];
        prj[1] = pnt_trans[1] / pnt_trans[2];
    }

    /// @brief pose derivativs for unit plane projection  
    /// @param pose - pose 
    /// @param point - point
    /// @return - projected point and derivative
    static Eigen::Vector2<ceres::Jet<T, 6>> df_dps(const T pose[6], const T point[3]) {
        static const T one = T(1.0);
        static const T zero = T(0.0);
        
        Eigen::Vector2<ceres::Jet<T, 6>> result; 

        auto rot_res = Rotation<T>::df(pose, point);

        const T inv_2 = T(1.0) / (rot_res.Rpt[2] + pose[5]);
        const T _0_by_2 = (rot_res.Rpt[0] + pose[3]) * inv_2;
        const T _1_by_2 = (rot_res.Rpt[1] + pose[4]) * inv_2;

        result[0].a = _0_by_2;
        result[1].a = _1_by_2;

        result[0].v[0] = (rot_res.df_daa[0] - _0_by_2 * rot_res.df_daa[2]) * inv_2;
        result[1].v[0] = (rot_res.df_daa[1] - _1_by_2 * rot_res.df_daa[2]) * inv_2;

        result[0].v[1] = (rot_res.df_daa[3] - _0_by_2 * rot_res.df_daa[5]) * inv_2;
        result[1].v[1] = (rot_res.df_daa[4] - _1_by_2 * rot_res.df_daa[5]) * inv_2;

        result[0].v[2] = (rot_res.df_daa[6] - _0_by_2 * rot_res.df_daa[8]) * inv_2;
        result[1].v[2] = (rot_res.df_daa[7] - _1_by_2 * rot_res.df_daa[8]) * inv_2;

        result[0].v[3] = inv_2;
        result[1].v[3] = zero;

        result[0].v[4] = zero;
        result[1].v[4] = inv_2;

        result[0].v[5] = - _0_by_2 * inv_2;
        result[1].v[5] = - _1_by_2 * inv_2;
        
        return result;
    }
};