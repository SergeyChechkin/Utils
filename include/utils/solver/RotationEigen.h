#pragma once

#include <Eigen/Core>

// so3 rotation for very small roatation angle 
template<typename T>
class Rotation_min{ 
public:
    static Eigen::Vector<T, 3> f(const Eigen::Vector<T, 3>& angle_axis, const Eigen::Vector<T, 3>& pnt) {   
        Eigen::Vector<T, 3> result;
        // (I + aa^) * pt
        result[0] = pnt[0] + angle_axis[1] * pnt[2] - angle_axis[2] * pnt[1];
        result[1] = pnt[1] + angle_axis[2] * pnt[0] - angle_axis[0] * pnt[2];
        result[2] = pnt[2] + angle_axis[0] * pnt[1] - angle_axis[1] * pnt[0];

        return result; 
    }

    static Eigen::Matrix<T, 3, 3> df_daa(const Eigen::Vector<T, 3>& angle_axis, const Eigen::Vector<T, 3>& pnt) {    
        static const T zero = T(0.0);
        Eigen::Matrix<T, 3, 3> result;
        // -pt^ 
        result << zero, pt[2], -pt[1], 
                  -pt[2], zero, pt[0], 
                  pt[1], -pt[0], zero;

        return result;
    }

    static Eigen::Matrix<T, 3, 3> df_dp(const Eigen::Vector<T, 3>& angle_axis, const Eigen::Vector<T, 3>& pnt) {    
        static const T one = T(1.0);
        Eigen::Matrix<T, 3, 3> result;
        // I + aa^
        result << one, -angle_axis[2], angle_axis[1], 
                  angle_axis[2], one, -angle_axis[0], 
                  -angle_axis[1], angle_axis[0], one;

        return result;
    }
};

template<typename T>
class Rotation { 
private:
    // precomputed rotation parameters
    template<typename T>
    struct AngleAxis {
        T axis[3];
        T theta;
        T sintheta;
        T costheta;
    };

    
public:
    static Eigen::Vector<T, 3> f(const Eigen::Vector<T, 3>& angle_axis, const Eigen::Vector<T, 3>& pnt) {
    }

    static Eigen::Matrix<T, 3, 3> df_daa(const Eigen::Vector<T, 3>& angle_axis, const Eigen::Vector<T, 3>& pnt) {
    }

    static Eigen::Matrix<T, 3, 3> df_dp(const Eigen::Vector<T, 3>& angle_axis, const Eigen::Vector<T, 3>& pnt) {
    }
};