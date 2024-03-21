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
struct MinFeature2D {
    MinFeature2D()
    : location_(Eigen::Vector2f::Zero())
    , gradient_(Eigen::Vector2f::Zero())
    {

    }
    
    Eigen::Vector2f location_;
    Eigen::Vector2f gradient_;  
};

}
