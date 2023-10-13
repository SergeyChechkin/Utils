#pragma once

#include <ceres/rotation.h>
#include <Eigen/Geometry>

Eigen::Vector<double, 6> Convert(const Eigen::Isometry3d& pose);
Eigen::Isometry3d Convert(const Eigen::Vector<double, 6>& pose);