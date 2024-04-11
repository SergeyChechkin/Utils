/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Core>

namespace lib::features {

// Minimal feature in image space. Location and gradient.
//
// Full image feature: location and information matrix. Practically a normal distribution.
// Assumption: Edge feature.
// We have only one principal component of info mat - gradient. 
template<typename T>
struct MinFeature2D_ {
    using PointT = Eigen::Vector2<T>; 
    MinFeature2D_()
    : location_(PointT::Zero())
    , gradient_(PointT::Zero())
    //, responce_(0)
    {

    }

    PointT location_;
    PointT gradient_;  
    //T responce_;
};

template<typename T>
struct MinFeature3D_ {
    using PointT = Eigen::Vector3<T>; 
    using VectT = Eigen::Vector3<T>; 
    using UnitVectT = Eigen::Vector3<T>; 

    MinFeature3D_() 
    : location_(PointT::Zero())
    , normal_(UnitVectT::Zero())
    , edge_(UnitVectT::Zero())
    {
    }

    PointT location_;
    UnitVectT normal_;      // Surfase normal  
    UnitVectT edge_;        // Edge line on the surfase plane
};

using MinFeature2D = MinFeature2D_<float>;
using MinFeature3D = MinFeature3D_<float>;

}
