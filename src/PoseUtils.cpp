#include "utils/PoseUtils.h"

Eigen::Vector<double, 6> Convert(const Eigen::Isometry3d& pose) {
    Eigen::Vector<double, 6> result;
    //Eigen::Matrix3d rot_mat = pose.rotation();
    //ceres::RotationMatrixToAngleAxis(rot_mat.data(), result.data());
    Eigen::AngleAxisd rot_aa(pose.rotation());
    result[0] = rot_aa.axis().x() * rot_aa.angle();
    result[1] = rot_aa.axis().y() * rot_aa.angle();
    result[2] = rot_aa.axis().z() * rot_aa.angle();
    result[3] = pose.translation()[0];
    result[4] = pose.translation()[1];
    result[5] = pose.translation()[2];
    return result;
}

Eigen::Isometry3d Convert(const Eigen::Vector<double, 6>& pose) {
    Eigen::Matrix3d rot_mat;
    ceres::AngleAxisToRotationMatrix(pose.data(), rot_mat.data());

    Eigen::Isometry3d result;
    result.linear() = rot_mat;
    result.translation() = pose.block<3,1>(3,0);

    return result;
}