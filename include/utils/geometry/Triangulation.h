#pragma once

#include <Eigen/Geometry>

/// @brief Triangulation from two unit plane correspondences, return depths for both rays.
/// Intersection is a segment, perpendicular to both rays, not a single point. 
/// To compute points: just multiply rays by corespondent depths.  
/// @param point0 - world frame observation
/// @param T - second frame pose
/// @param point1 - second frame observation
/// @param d0 - first observation depth
/// @param d1 - second observation depth
/// @return - true, if successful (rays not parallel) 
bool TriangulatePointDepths(
    const Eigen::Vector2d& point_0,
    const Eigen::Isometry3d& T,
    const Eigen::Vector2d& point_1,
    double& d0, 
    double& d1);