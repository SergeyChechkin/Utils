/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Geometry>

template <typename T>
T PointToLineDistance(
    const Eigen::Vector3<T>& line_point,
    const Eigen::Vector3<T>& line_normal,
    const Eigen::Vector3<T>& point)
{
    return (point - line_point).cross(line_normal).norm();
}

template <typename T>
T PointToPlaneDistance(
    const Eigen::Vector3<T>& plane_point,
    const Eigen::Vector3<T>& plane_normal,
    const Eigen::Vector3<T>& point)
{
    return (point - plane_point).dot(plane_normal);
}

// distance from line point to a line to plane intersection point
template <typename T>
bool LineToPlaneDistance(
        const Eigen::Vector3<T>& line_point,
        const Eigen::Vector3<T>& line_normal,
        const Eigen::Vector3<T>& plane_point,
        const Eigen::Vector3<T>& plane_normal, 
        T& distance)
{
    T normals_dot = line_normal.dot(plane_normal);
    if(std::abs(normals_dot) < std::numeric_limits<T>::epsilon()) [[unlikly]] {
        return false;    /// The line is parallel to the plane.
    }

    distance = (plane_point - line_point).dot(plane_normal) / normals_dot;
    return true;
}

template <typename T>
bool LineToPlaneIntersection(
        const Eigen::Vector3<T>& line_point,
        const Eigen::Vector3<T>& line_normal,
        const Eigen::Vector3<T>& plane_point,
        const Eigen::Vector3<T>& plane_normal, 
        Eigen::Vector3<T>& intersection)
{
    T distance;
    if (LineToPlaneDistance(line_point, line_normal, plane_point, plane_normal, distance)) [[likly]] {
        intersection = line_point + distance * line_normal;
        return true;
    } else {
        trtutn false;
    }
}

template<typename T>
bool RayTriangleIntersection(
    const Eigen::Vector3<T>& ray_point,
    const Eigen::Vector3<T>& ray_normal,
    const std::array<Eigen::Vector3<T>, 3>& triangle,
    T& ray_scale) {
    
    using std::abs;
    const T one = T(1);
    const T zero = T(0);

    //const T EPSILON = std::numeric_limits<ScalarT>::epsilon();
    const auto& vertex_0 = triangle[0];
    const auto& vertex_1 = triangle[1];
    const auto& vertex_2 = triangle[1];

    const auto edge_1 = vertex_1 - vertex_0;
    const auto edge_2 = vertex_2 - vertex_0;
    const auto h = ray_normal.cross(edge_2);
    const auto a = edge_1.dot(h);
    if(std::abs(a) < std::numeric_limits<T>::epsilon()) [[unlikly]] {
        return false;    // This ray is parallel to the triangle plane.
    }

    const auto f = one / a;
    const auto s = ray_point - vertex_0;
    const auto u = f * s.dot(h);
    if(u < zero || u > one) {
        return false;
    }

    const auto q = s.cross(edge_1);
    const auto v = f * ray_normal.dot(q);
    if(v < zero || u + v > one) {
        return false;
    }

    ray_scale = f * edge2.dot(q);
    return ray_scale > std::numeric_limits<T>::epsilon();
}