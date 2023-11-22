/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#include "utils/features/HomographySolver.h"
#include "utils/geometry/Triangulation.h"

#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <glog/logging.h>

bool HomographySolver::Solve(
    const std::vector<Eigen::Vector2d>& prev, 
    const std::vector<Eigen::Vector2d>& next, 
    Eigen::Isometry3d& pose, 
    const Cofiguration& config) 
{
    CHECK_GE(prev.size(), 5);
    CHECK_EQ(prev.size(), prev.size());

    Eigen::Matrix3d E;
    ComputeEssentialMatrix(prev, next, E, config);

    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d t;
    DecomposeEssentialMatrix(E, R1, R2, t);

    Eigen::Matrix3d Rs[4] = {R1, R1, R2, R2};
    Eigen::Vector3d ts[4] = {t, -t, t, -t};
    size_t res_counts = 0;

    for(size_t j = 0; j < 4; ++j) {
        Eigen::Isometry3d T;
        T.linear() = Rs[j];
        T.translation() = ts[j];
        size_t counts = 0;

        size_t size = std::min(config.depth_test_itr_, prev.size());
        for(size_t i = 0; i < size; ++i) {
            double d_prev, d_next;
            if (TriangulatePointDepths(prev[i], T, next[i], d_prev, d_next) && d_prev > 0 && d_next > 0)  {
                ++counts;
            }
        }

        if (res_counts < counts) {
            res_counts = counts;
            pose = T;
        }
    }

    return res_counts >= 5; 
}

bool HomographySolver::Solve(
    const std::vector<Eigen::Vector2d>& prev, 
    const std::vector<Eigen::Vector2d>& next, 
    Eigen::Isometry3d& pose,
    std::vector<double>& depths, 
    const Cofiguration& config)
{
    CHECK_GE(prev.size(), 5);
    CHECK_EQ(prev.size(), prev.size());

    Eigen::Matrix3d E;
    ComputeEssentialMatrix(prev, next, E, config);

    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d t;
    DecomposeEssentialMatrix(E, R1, R2, t);

    Eigen::Matrix3d Rs[4] = {R1, R1, R2, R2};
    Eigen::Vector3d ts[4] = {t, -t, t, -t};
    size_t res_counts = 0;


    for(size_t j = 0; j < 4; ++j) {
        Eigen::Isometry3d T;
        T.linear() = Rs[j];
        T.translation() = ts[j];
        size_t counts = 0;
        std::vector<double> dpts(prev.size(), 1);

        for(size_t i = 0; i < prev.size(); ++i) {
            double d_prev, d_next;
            if (TriangulatePointDepths(prev[i], T, next[i], d_prev, d_next) && d_prev > 0 && d_next > 0)  {
                ++counts;
                dpts[i] = d_prev;
            }
        }

        if (res_counts < counts) {
            res_counts = counts;
            pose = T;
            depths = dpts;
        }
    }

    return res_counts >= 5; 
}

inline std::vector<cv::Point2f> Convert(const std::vector<Eigen::Vector2d>& src) {
    std::vector<cv::Point2f> result(src.size());

    for(size_t i = 0; i < src.size(); ++i) {
       const auto& src_p = src[i];
       auto& dst_p = result[i];
       dst_p.x = src_p[0];
       dst_p.y = src_p[1];
    }

    return result;
}

size_t HomographySolver::ComputeEssentialMatrix(
    const std::vector<Eigen::Vector2d>& prev, 
    const std::vector<Eigen::Vector2d>& next, 
    Eigen::Matrix3d& E, 
    const Cofiguration& config) 
{
    const std::vector<cv::Point2f> prev_cv = Convert(prev); 
    const std::vector<cv::Point2f> next_cv = Convert(next);
    std::vector<uchar> mask; 

    auto E_cv = cv::findEssentialMat(
        prev_cv, next_cv, 
        config.focal_, 
        config.pp_, 
        config.method_, 
        config.prob_, 
        config.threshold_, 
        config.max_iters_,
        mask);
    cv::cv2eigen(E_cv, E);

    return std::accumulate(mask.begin(), mask.end(), 0);
}

void HomographySolver::DecomposeEssentialMatrix(
    const Eigen::Matrix3d& E,
    Eigen::Matrix3d& R1,
    Eigen::Matrix3d& R2,
    Eigen::Vector3d& t) 
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV().transpose();

    //if (U.determinant() < 0) { U *= -1;}
    //if (V.determinant() < 0) { V *= -1;}

    Eigen::Matrix3d W;
    W << 0, 1, 0, -1, 0, 0, 0, 0, 1;

    R1 = (U * W * V).transpose();
    R2 = (U * W.transpose() * V).transpose();
    t = U.col(2).normalized();
}

class Homography_q_CF {
public:
    template<typename T>
    bool operator()(const T q[4], const T t[3], T residuals[1]) const {

        Eigen::Matrix3<T> R;
        ceres::QuaternionToRotation(q, R.data());

        Eigen::Matrix3<T> t_x;
        t_x << T(0), t[2], -t[1], -t[2], T(0), t[0], t[1], -t[0], T(0);

        const Eigen::Matrix3<T> E = t_x * R;

        const Eigen::Vector3<T> pt_0 = plane_point_0_.homogeneous().cast<T>();
        const Eigen::Vector3<T> pt_1 = plane_point_1_.homogeneous().cast<T>();

        residuals[0] = pt_1.transpose() * E * pt_0;

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& plane_point_0, const Eigen::Vector2d& plane_point_1) {
            return new ceres::AutoDiffCostFunction<Homography_q_CF, 1, 4, 3>(
                new Homography_q_CF(plane_point_0, plane_point_1));
        }

    static ceres::CostFunction* Create(Homography_q_CF* cf) {
            return new ceres::AutoDiffCostFunction<Homography_q_CF, 1, 4, 3>(cf);
        }

    Homography_q_CF(const Eigen::Vector2d& plane_point_0, const Eigen::Vector2d& plane_point_1) 
        : plane_point_0_(plane_point_0), plane_point_1_(plane_point_1) {}
private:
    Eigen::Vector2d plane_point_0_;
    Eigen::Vector2d plane_point_1_;
};

bool HomographySolver::Solve_Ceres_qt(
    const std::vector<Eigen::Vector2d>& prev, 
    const std::vector<Eigen::Vector2d>& next, 
    Eigen::Vector4d& q,  
    Eigen::Vector3d& t) 
{

    CHECK_EQ(prev.size(), next.size());

    ceres::Problem problem;

    const double loss_threshold = 2.0 / 465;
    ceres::LossFunction* lf = new ceres::CauchyLoss(loss_threshold);

    for(size_t i = 0; i < prev.size(); ++i) {
        auto* cf = Homography_q_CF::Create(prev[i], next[i]);
        problem.AddResidualBlock(cf, lf, q.data(), t.data());
    }

    problem.SetManifold(q.data(), new ceres::QuaternionManifold);
    problem.SetManifold(t.data(), new ceres::SphereManifold<3>);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = std::thread::hardware_concurrency();
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cerr << summary.BriefReport() << std::endl; 
    std::cerr << summary.message << std::endl; 

    return summary.IsSolutionUsable();
}

class Homography_aa_CF {
public:
    template<typename T>
    bool operator()(const T aa[3], const T t[3], T residuals[1]) const {

        Eigen::Matrix3<T> R;
        ceres::AngleAxisToRotationMatrix(aa, R.data());

        Eigen::Matrix3<T> t_x;
        t_x << T(0), t[2], -t[1], -t[2], T(0), t[0], t[1], -t[0], T(0);

        const Eigen::Matrix3<T> E = t_x * R;

        const Eigen::Vector3<T> pt_0 = plane_point_0_.homogeneous().cast<T>();
        const Eigen::Vector3<T> pt_1 = plane_point_1_.homogeneous().cast<T>();

        residuals[0] = pt_1.transpose() * E * pt_0;

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& plane_point_0, const Eigen::Vector2d& plane_point_1) {
            return new ceres::AutoDiffCostFunction<Homography_aa_CF, 1, 3, 3>(
                new Homography_aa_CF(plane_point_0, plane_point_1));
        }

    static ceres::CostFunction* Create(Homography_aa_CF* cf) {
            return new ceres::AutoDiffCostFunction<Homography_aa_CF, 1, 3, 3>(cf);
        }

    Homography_aa_CF(const Eigen::Vector2d& plane_point_0, const Eigen::Vector2d& plane_point_1) 
        : plane_point_0_(plane_point_0), plane_point_1_(plane_point_1) {}
private:
    Eigen::Vector2d plane_point_0_;
    Eigen::Vector2d plane_point_1_;
};

bool HomographySolver::Solve_Ceres(
    const std::vector<Eigen::Vector2d>& prev, 
    const std::vector<Eigen::Vector2d>& next, 
    Eigen::Vector3d& aa,  
    Eigen::Vector3d& t) 
{
    CHECK_EQ(prev.size(), next.size());

    ceres::Problem problem;

    const double loss_threshold = 2.0 / 465;
    ceres::LossFunction* lf = new ceres::CauchyLoss(loss_threshold);

    for(size_t i = 0; i < prev.size(); ++i) {
        auto* cf = Homography_aa_CF::Create(prev[i], next[i]);
        problem.AddResidualBlock(cf, lf, aa.data(), t.data());
    }

    problem.SetManifold(t.data(), new ceres::SphereManifold<3>);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = std::thread::hardware_concurrency();
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ///
    aa = -aa;

    std::cerr << summary.BriefReport() << std::endl; 
    std::cerr << summary.message << std::endl; 

    return summary.IsSolutionUsable();
}
