# pragma once

#include <Eigen/Core>

namespace InverseDepth {

    template<typename T>
    class PerspectiveReprojection {
    public:
        static Eigen::Vector<T, 3> f(const Eigen::Vector<T, 2>& pnt, T inv_depth) {
            Eigen::Vector<T, 3> result;
            
            result[2] = T(1.0) / inv_depth;
            result[0] = pnt[0] * result[2];
            result[1] = pnt[1] * result[2];

            return result;
        }

        static Eigen::Matrix<T, 1, 3> df_inv_d(const Eigen::Vector<T, 2>& pnt, T inv_depth) {
            Eigen::Matrix<T, 1, 3> result;

            result[2] = - T(1.0) / (inv_depth * inv_depth);
            result[0] = pnt[0] * result[2];
            result[1] = pnt[1] * result[2];

            return result;
        }

    };

    template<typename T>
    class Transformation {
    public:
        static Eigen::Vector3<T> f(const Eigen::Vector<T, 6>& pose, const Eigen::Vector3<T>& pnt) {
            Eigen::Vector3<T> result;

            Rotation<T>::f(pose.data(), pnt.data(), result.data());
            result[0] += pose[3];
            result[1] += pose[4];
            result[2] += pose[5];

            return result;
        }

        static Eigen::Matrix<T, 3, 3> df_pnt(const T pose[6], const T src[3]) {
            
        }

        static Eigen::Matrix<T, 6, 3> df_pose(const T pose[6], const T src[3]) {
            
        }

    };
    
    template<typename T>
    class ReprojectTransform {
    public:
        static Eigen::Vector<T, 3> f(const Eigen::Vector<T, 2>& pnt, T inv_depth, const Eigen::Vector<T, 6>& pose) {
            return InverseDepth::Transformation<T>::f(pose, PerspectiveReprojection(pnt, inv_depth));
        }

        static Eigen::Matrix<T, 1, 3> df_inv_d(const Eigen::Vector<T, 2>& pnt, T inv_depth, const Eigen::Vector<T, 6>& pose) {
            //return InverseDepth::Transformation<T>::df_
        }
    };
} 
