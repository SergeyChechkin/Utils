// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "Rotation.h"
#include <ceres/jet.h>
#include <ceres/rotation.h>

template<typename T>
class InverseDepthUnitPlaneTransformation {
public:
    static void f(const T pose[6], const T src[2], const T& inv_depth, T dst[2]) {
        const T src_3d[3];
        src_3d[0] = src[0] / inv_depth;
        src_3d[1] = src[1] / inv_depth;
        src_3d[2] = T(1.0) / inv_depth;

        T dst_3d[2];
        Transformation<T>::f(pose, src_3d, dst_3d);

        dst[0] = dst_3d[0] / dst_3d[2];
        dst[1] = dst_3d[1] / dst_3d[2];
    }


    static Eigen::Vector2<ceres::Jet<T, 9>> df_cj(const T pose[6], const T src[2], const T& inv_depth) {
        using JetT = ceres::Jet<double, 9>;
        
        Eigen::Vector<JetT, 6> pose_j;
        for(int i = 0; i < 6; ++i)  {
            pose_j[i] = JetT(pose[i], i);
        }

        Eigen::Vector2<JetT> src_j;
        for(int i = 0; i < 2; ++i)  {
            src_j[i] = JetT(src[i], i + 6);
        }

        JetT inv_depth_j(inv_depth, 8);

        Eigen::Vector2<JetT> res_j;
        InverseDepthUnitPlaneTransformation<JetT>::f(pose_j.data(), src_j.data(), inv_depth_j, res_j.data());

        return res_j;
    }
};