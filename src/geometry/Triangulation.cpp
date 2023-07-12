#include "utils/geometry/Triangulation.h"

bool TriangulatePointDepths(
    const Eigen::Vector2d& point_0,
    const Eigen::Isometry3d& T,
    const Eigen::Vector2d& point_1,
    double& d0, 
    double& d1) 
{
    const Eigen::Vector3d r0 = point_0.homogeneous();               // first ray in world framy (by default)
    const Eigen::Vector3d r1 = T.linear() * point_1.homogeneous();  // second ray in world frame
    const Eigen::Vector3d r2 = r0.cross(r1);                        // cross product of rays

    if (r2.squaredNorm() < std::numeric_limits<double>::epsilon())
        return false;                                               // parallel rays

    const Eigen::Vector3d p1 = T.translation();                     // second ray origin

    Eigen::Matrix3d A;
    A.col(0) = r0;
    A.col(1) = -r1;
    A.col(2) = r2;

    const Eigen::Vector3d d = A.inverse() * p1;
    d0 = d[0];
    d1 = d[1];

    return true;
}