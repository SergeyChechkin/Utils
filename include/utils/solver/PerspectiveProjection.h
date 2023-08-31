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
    /// @brief tranformation and unit plane projection (f = 1) 
    /// @param pose - pose 
    /// @param pnt - 3D point 
    /// @param prj - 2D projection
    static void f(const T pose[6], const T pnt[3], T prj[2]) {
        T pnt_trans[3];
        Transformation<T>::f(pose, pnt, pnt_trans);

        prj[0] = pnt_trans[0] / pnt_trans[2];
        prj[1] = pnt_trans[1] / pnt_trans[2];
    }

    /// @brief tranformation derivativs for unit plane projection (f = 1)  
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

    static Eigen::Vector2<T> f(T fcl, const Eigen::Vector3<T>& pnt) {
        Eigen::Vector2<T> result;

        const T f_by_pnt_2 = fcl / pnt[2];
        result[0] = f_by_pnt_2 * pnt[0];
        result[1] = f_by_pnt_2 * pnt[1];

        return result;
    }

    static Eigen::Vector2<T> f(const Eigen::Vector<T, 6>& pose, T fcl, const Eigen::Vector3<T>& pnt) {
        const auto trans_pnt = Transformation<T>::f(pose, pnt);
        return f(fcl, trans_pnt);
    }

    static Eigen::Matrix<T, 2, 3> df_dpnt(T fcl, const Eigen::Vector3<T>& pnt) {
        static const T zero = T(0.0);
    
        Eigen::Matrix<T, 2, 3> result;

        const T inv_z = T(1.0) / pnt[2];
        const T f_by_z = fcl * inv_z;
        const T f_by_z_sqr = -f_by_z * inv_z;  
        
        result.row(0) << f_by_z, zero, pnt[0] * f_by_z_sqr;
        result.row(1) << zero, f_by_z, pnt[1] * f_by_z_sqr;

        return result;
    }

    /// zero transformation derivative
    static Eigen::Matrix<T, 2, 6> df_dps_zero(T fcl, const Eigen::Vector3<T>& pnt) {
        return df_dpnt(fcl, pnt) * Transformation<T>::df_dps_zero(pnt.data());
    }
};