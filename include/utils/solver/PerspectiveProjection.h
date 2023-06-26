/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "Functions.h"
#include <Eigen/Core>
#include <ceres/jet.h>
#include <cmath>

template<typename T>
class PerspectiveProjection{ 
public:
    void f(const T pnt[3], T prj[2]) {
        prj[0] = pnt[0] / pnt[2];
        prj[1] = pnt[1] / pnt[2];
    }
};