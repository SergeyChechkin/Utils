/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "Functions.h"
#include <Eigen/Core>
#include <ceres/jet.h>
#include <cmath>

/// @brief so3 rotation for very small roatation angle 
/// @tparam T - scalar type (float, double or ceres::Jet)
template<typename T>
class Rotation_min{ 
public:
    /// @brief 3D point rotation function
    /// @param angle_axis - rotation    
    /// @param pt - point 
    /// @param result - rotated point 
    static void f(const T angle_axis[3], const T pt[3], T result[3]) {   
        // (I + aa^) * pt
        result[0] = pt[0] + angle_axis[1] * pt[2] - angle_axis[2] * pt[1];
        result[1] = pt[1] + angle_axis[2] * pt[0] - angle_axis[0] * pt[2];
        result[2] = pt[2] + angle_axis[0] * pt[1] - angle_axis[1] * pt[0]; 
    }

    /// @brief 3D point rotation partial derivative by rotation
    /// @param pt - point
    /// @param result - partial derivative
    static void df_daa(const T pt[3], T result[9]) {    
        static const T zero = T(0.0);
        // -pt^ 
        result[0] =  zero;
        result[1] =  -pt[2];
        result[2] =  pt[1];
        
        result[3] =  pt[2];
        result[4] =  zero;
        result[5] =  -pt[0];
        
        result[6] =  -pt[1];
        result[7] =  pt[0];
        result[8] =  zero;
    }

    /// @brief 3D point rotation partial derivative by point
    /// @param angle_axis - rotation
    /// @param result - partial derivative 
    static void df_dp(const T angle_axis[3], T result[9]) {    
        static const T one = T(1.0);
        // I + aa^
        result[0] = one;
        result[1] = angle_axis[2];
        result[2] = - angle_axis[1];

        result[3] = - angle_axis[2];
        result[4] = one;
        result[5] = angle_axis[0];

        result[6] = angle_axis[1];
        result[7] = - angle_axis[0];
        result[8] = one;
    }
};

/// @brief precomputed rotation parameters
/// @tparam T - scalar type
template<typename T>
struct AngleAxis {
        T axis[3];
        T theta;
        T sintheta;
        T costheta;
    };


template<typename T>
class Rotation_full{ 
public:
    /// @brief 3D point rotation function
    /// @param aa - rotation 
    /// @param pt - point 
    /// @param f - rotated point  
    static void f(const AngleAxis<T> aa, const T pt[3], T f[3]) {
        static const T one = T(1.0);
        const T a_cross_pt[3] = {aa.axis[1] * pt[2] - aa.axis[2] * pt[1],
                                aa.axis[2] * pt[0] - aa.axis[0] * pt[2],
                                aa.axis[0] * pt[1] - aa.axis[1] * pt[0]};

        // a_t * pt * (1-cos(theta))
        const T one_costh = (one - aa.costheta);
        const T tmp = (aa.axis[0] * pt[0] + aa.axis[1] * pt[1] + aa.axis[2] * pt[2]) * one_costh;

        f[0] = pt[0] * aa.costheta + a_cross_pt[0] * aa.sintheta + aa.axis[0] * tmp;
        f[1] = pt[1] * aa.costheta + a_cross_pt[1] * aa.sintheta + aa.axis[1] * tmp;
        f[2] = pt[2] * aa.costheta + a_cross_pt[2] * aa.sintheta + aa.axis[2] * tmp;
    }

    /// @brief 3D point rotation partial derivative by rotation
    /// @param aa - rotation
    /// @param Rp - rotated point
    /// @param df_daa - partial derivative 
    static void df_daa(const AngleAxis<T> aa, const T Rp[3], T df_daa[9]) {
        static const T one = T(1.0);
        // -(Rp)^J 
        const T c0 = aa.sintheta / aa.theta;
        const T c1 = one - c0;
        const T c2 = (one - aa.costheta) / aa.theta;

        const T w00 = c1 * aa.axis[0] * aa.axis[0];
        const T w01 = c1 * aa.axis[0] * aa.axis[1];
        const T w02 = c1 * aa.axis[0] * aa.axis[2];
        const T w11 = c1 * aa.axis[1] * aa.axis[1];
        const T w12 = c1 * aa.axis[1] * aa.axis[2];
        const T w22 = c1 * aa.axis[2] * aa.axis[2];

        const T c2w0 = c2 * aa.axis[0];
        const T c2w1 = c2 * aa.axis[1];
        const T c2w2 = c2 * aa.axis[2];
        
        const T J[9] = {c0 + w00, w01 + c2w2, w02 - c2w1, 
                        w01 - c2w2, c0 + w11, w12 + c2w0,
                        w02 + c2w1, w12 - c2w0, c0 + w22};
        
        CrossProduct(J, Rp, df_daa);
        CrossProduct(J + 3, Rp, df_daa + 3);
        CrossProduct(J + 6, Rp, df_daa + 6);
    }


    static void df_dp(const AngleAxis<T> aa, T df_dp[9]) {
        static const T one = T(1.0);
        const T one_costh = (one - aa.costheta);
        const T dtmp_dpt[3] = {aa.axis[0] * one_costh, aa.axis[1] * one_costh, aa.axis[2] * one_costh};

        df_dp[0] = aa.costheta + aa.axis[0] * dtmp_dpt[0];
        df_dp[1] = aa.axis[2] * aa.sintheta + aa.axis[1] * dtmp_dpt[0];
        df_dp[2] = - aa.axis[1] * aa.sintheta + aa.axis[2] * dtmp_dpt[0];

        df_dp[3] = - aa.axis[2] * aa.sintheta + aa.axis[0] * dtmp_dpt[1];
        df_dp[4] = aa.costheta + aa.axis[1] * dtmp_dpt[1];
        df_dp[5] = aa.axis[0] * aa.sintheta + aa.axis[2] * dtmp_dpt[1];

        df_dp[6] = aa.axis[1] * aa.sintheta + aa.axis[0] * dtmp_dpt[2];
        df_dp[7] = - aa.axis[0] * aa.sintheta + aa.axis[1] * dtmp_dpt[2];
        df_dp[8] = aa.costheta + aa.axis[2] * dtmp_dpt[2];
    }
};


/// @brief so3 rotation  
/// @tparam T - scalar type or ceres::Jet
template<typename T>
class Rotation{ 
public:
    static void f(const T angle_axis[3], const T pt[3], T Rpt[3]) {  
        using std::hypot;
        using std::sin;
        using std::cos;
        using std::fpclassify;
        static const T one = T(1.0);
        
        AngleAxis<T> aa;
        aa.theta = hypot(angle_axis[0], angle_axis[1], angle_axis[2]);

        if (FP_ZERO == fpclassify(aa.theta)) {
            Rotation_min<T>::f(angle_axis, pt, Rpt); 
            return; 
        }
        
        aa.costheta = cos(aa.theta);
        aa.sintheta = sin(aa.theta);        
        const T theta_inverse = one / aa.theta;
        aa.axis[0] = angle_axis[0] * theta_inverse;
        aa.axis[1] = angle_axis[1] * theta_inverse;
        aa.axis[2] = angle_axis[2] * theta_inverse;

        Rotation_full<T>::f(aa, pt, Rpt);
    }

    static Eigen::Vector<T, 3> f(const Eigen::Vector<T, 3>& angle_axis, const Eigen::Vector<T, 3>& pnt) {  
        using std::hypot;
        using std::sin;
        using std::cos;
        using std::fpclassify;
        static const T one = T(1.0);
        
        AngleAxis<T> aa;
        aa.theta = hypot(angle_axis[0], angle_axis[1], angle_axis[2]);

        Eigen::Vector<T, 3> result;     
        if (FP_ZERO == fpclassify(aa.theta)) {
            Rotation_min<T>::f(angle_axis.data(), pnt.data(), result.data()); 
            return result; 
        }
        
        aa.costheta = cos(aa.theta);
        aa.sintheta = sin(aa.theta);        
        const T theta_inverse = one / aa.theta;
        aa.axis[0] = angle_axis[0] * theta_inverse;
        aa.axis[1] = angle_axis[1] * theta_inverse;
        aa.axis[2] = angle_axis[2] * theta_inverse;

        Rotation_full<T>::f(aa, pnt.data(), result.data());
        return result;
    }

    static std::vector<Eigen::Vector3<T>> f(const T angle_axis[3], std::vector<Eigen::Vector3<T>>& pts) {
        using std::hypot;
        using std::sin;
        using std::cos;
        using std::fpclassify;
        static const T one = T(1.0);
        std::vector<Eigen::Vector3<T>> result(pts.size());
        AngleAxis<T> aa;
        aa.theta = hypot(angle_axis[0], angle_axis[1], angle_axis[2]);

        if (FP_ZERO == fpclassify(aa.theta)) {
            for(size_t i = 0; i < pts.size(); ++i) {
                Rotation_min<T>::f(aa, pts[i * 3], result[i].data());
            }
            return result; 
        }
        
        aa.costheta = cos(aa.theta);
        aa.sintheta = sin(aa.theta);        
        const T theta_inverse = one / aa.theta;
        aa.axis[0] = angle_axis[0] * theta_inverse;
        aa.axis[1] = angle_axis[1] * theta_inverse;
        aa.axis[2] = angle_axis[2] * theta_inverse;

        for(size_t i = 0; i < pts.size(); ++i) {
            Rotation_full<T>::f(aa, pts[i].data(), result[i].data());
        }

        return result;
    }

    struct RotatedPoint {
       T Rpt[3];        // rotated point
       T df_daa[9];     // Angle-axis parial derivativs 
       T df_dp[9];      // point parial derivativs
    };

    static RotatedPoint df(const T angle_axis[3], const T pt[3]) {
        using std::hypot;
        using std::sin;
        using std::cos;
        using std::fpclassify;
        static const T one = T(1.0);
        AngleAxis<T> aa;
        RotatedPoint result;
        aa.theta = hypot(angle_axis[0], angle_axis[1], angle_axis[2]);

        if (FP_ZERO == fpclassify(aa.theta)) {
            Rotation_min<T>::f(angle_axis, pt, result.Rpt); 
            Rotation_min<T>::df_daa(pt, result.df_daa); 
            Rotation_min<T>::df_dp(angle_axis, result.df_dp); 
            return result; 
        }

        aa.costheta = cos(aa.theta);
        aa.sintheta = sin(aa.theta);        
        const T theta_inverse = one / aa.theta;
        aa.axis[0] = angle_axis[0] * theta_inverse;
        aa.axis[1] = angle_axis[1] * theta_inverse;
        aa.axis[2] = angle_axis[2] * theta_inverse;

        Rotation_full<T>::f(aa, pt, result.Rpt);
        Rotation_full<T>::df_daa(aa, result.Rpt, result.df_daa);
        Rotation_full<T>::df_dp(aa, result.df_dp);

        return result;
    }

    static std::vector<RotatedPoint> df(const T angle_axis[3], std::vector<Eigen::Vector3<T>>& pts) {
        using std::hypot;
        using std::sin;
        using std::cos;
        using std::fpclassify;
        static const T one = T(1.0);

        std::vector<RotatedPoint> result(pts.size());
        AngleAxis<T> aa;
        aa.theta = hypot(angle_axis[0], angle_axis[1], angle_axis[2]);

        if (FP_ZERO == fpclassify(aa.theta)) {
            for(size_t i = 0; i < pts.size(); ++i) {
                RotatedPoint& r = result[i];
                Rotation_min<T>::f(angle_axis, pts[i].data(), r.Rpt); 
                Rotation_min<T>::df_daa(pts[i].data(), r.df_daa); 
                Rotation_min<T>::df_dp(angle_axis, r.df_dp); 
            }
            return result; 
        }

        aa.costheta = cos(aa.theta);
        aa.sintheta = sin(aa.theta);        
        const T theta_inverse = one / aa.theta;
        aa.axis[0] = angle_axis[0] * theta_inverse;
        aa.axis[1] = angle_axis[1] * theta_inverse;
        aa.axis[2] = angle_axis[2] * theta_inverse;

        for(size_t i = 0; i < pts.size(); ++i) {
            RotatedPoint& r = result[i];
            Rotation_full<T>::f(aa, pts[i].data(), r.Rpt);
            Rotation_full<T>::df_daa(aa, r.Rpt, r.df_daa);
            Rotation_full<T>::df_dp(aa, r.df_dp);
        }

        return result; 
    }
};

template<typename T>
class RotationUnitQuaternion{ 
public:
    static void f(const T q[4], const T pt[3], T result[3]) {   
        static const T two = T(2); 
        const T uv0 = two * (q[1] * pt[2] - q[2] * pt[1]);
        const T uv1 = two * (q[2] * pt[0] - q[0] * pt[2]);
        const T uv2 = two * (q[0] * pt[1] - q[1] * pt[0]);
        result[0] = pt[0] + q[3] * uv0 + q[1] * uv2 - q[2] * uv1;
        result[1] = pt[1] + q[3] * uv1 + q[2] * uv0 - q[0] * uv2;
        result[2] = pt[2] + q[3] * uv2 + q[0] * uv1 - q[1] * uv0;
    }

    static void df_dq(const T q[4], const T pt[3], T f[3], T df_dq[12]) {  
        static const T zero = T(0); 
        static const T one = T(1); 
        static const T two = T(2); 
        const T two_pt[3] = {two * pt[0], two * pt[1], two * pt[2]};

        const T uv0 = q[1] * two_pt[2] - q[2] * two_pt[1];
        const T uv1 = q[2] * two_pt[0] - q[0] * two_pt[2];
        const T uv2 = q[0] * two_pt[1] - q[1] * two_pt[0];
        f[0] = pt[0] + q[3] * uv0 + q[1] * uv2 - q[2] * uv1;
        f[1] = pt[1] + q[3] * uv1 + q[2] * uv0 - q[0] * uv2;
        f[2] = pt[2] + q[3] * uv2 + q[0] * uv1 - q[1] * uv0;

        df_dq[0] = q[1] * two_pt[1] + q[2] * two_pt[2];
        df_dq[1] = -q[3] * two_pt[2]  -uv2 - q[0] * two_pt[1];
        df_dq[2] = q[3] * two_pt[1] + uv1 - q[0] * two_pt[2];
        
        df_dq[3] = q[3] * two_pt[2] + uv2 - q[1] * two_pt[0];
        df_dq[4] = q[2] * two_pt[2] + q[0] * two_pt[0];
        df_dq[5] = -q[3] * two_pt[0] - uv0 - q[1] * two_pt[2];
        
        df_dq[6] = - q[3] * two_pt[1] - uv1 - q[2] * two_pt[0];
        df_dq[7] = q[3] * two_pt[0] + uv0 - q[2] * two_pt[1];        
        df_dq[8] = q[0] * two_pt[0] + q[1] * two_pt[1];
        
        df_dq[9] = uv0;
        df_dq[10] = uv1;
        df_dq[11] = uv2;
    }
};