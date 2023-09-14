/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#include "utils/solver/Rotation.h"
#include "utils/solver/Transformation.h"
#include "utils/solver/PerspectiveProjection.h"
#include "utils/solver/PnPSolver.h"
#include "utils/solver/HomographySolver.h"
#include "utils/geometry/Triangulation.h"

#include "utils/pipeline/Thread.h"
#include "utils/pipeline/SPSCQueue.h"
#include "utils/pipeline/ThreadPool.h"
#include "utils/pipeline/MemoryPool.h"
#include "utils/pipeline/PipelineNode.h"

#include "utils/PoseUtils.h"

#include <ceres/jet.h>
#include <gtest/gtest.h>
#include <random>

TEST(SolverUtils, RotationTest) { 
    const double aa[3] = {M_PI / 6, 0, 0};
    const double pt[3] = {1, 1, 1};
    
    auto result = Rotation<double>::df(aa, pt);

    // using ceres::Jet for comparosing 
    using JetT = ceres::Jet<double, 6>;
    Eigen::Vector3<JetT> pt_j, aa_j;
    for(int i = 0; i < 3; ++i)  {
        aa_j[i] = JetT(aa[i], i);
        pt_j[i] = JetT(pt[i], i+3);
    }

    Eigen::Vector3<JetT> res_j;
    Rotation<JetT>::f(aa_j.data(), pt_j.data(), res_j.data());

    ASSERT_DOUBLE_EQ(res_j[0].a, result.Rpt[0]);
    ASSERT_DOUBLE_EQ(res_j[1].a, result.Rpt[1]);
    ASSERT_DOUBLE_EQ(res_j[2].a, result.Rpt[2]);

    ASSERT_DOUBLE_EQ(res_j[0].v[0], result.df_daa[0]);
    ASSERT_DOUBLE_EQ(res_j[1].v[0], result.df_daa[1]);
    ASSERT_DOUBLE_EQ(res_j[2].v[0], result.df_daa[2]);

    ASSERT_DOUBLE_EQ(res_j[0].v[1], result.df_daa[3]);
    ASSERT_DOUBLE_EQ(res_j[1].v[1], result.df_daa[4]);
    ASSERT_DOUBLE_EQ(res_j[2].v[1], result.df_daa[5]);

    ASSERT_DOUBLE_EQ(res_j[0].v[2], result.df_daa[6]);
    ASSERT_DOUBLE_EQ(res_j[1].v[2], result.df_daa[7]);
    ASSERT_DOUBLE_EQ(res_j[2].v[2], result.df_daa[8]);

    ASSERT_DOUBLE_EQ(res_j[0].v[3], result.df_dp[0]);
    ASSERT_DOUBLE_EQ(res_j[1].v[3], result.df_dp[1]);
    ASSERT_DOUBLE_EQ(res_j[2].v[3], result.df_dp[2]);

    ASSERT_DOUBLE_EQ(res_j[0].v[4], result.df_dp[3]);
    ASSERT_DOUBLE_EQ(res_j[1].v[4], result.df_dp[4]);
    ASSERT_DOUBLE_EQ(res_j[2].v[4], result.df_dp[5]);

    ASSERT_DOUBLE_EQ(res_j[0].v[5], result.df_dp[6]);
    ASSERT_DOUBLE_EQ(res_j[1].v[5], result.df_dp[7]);
    ASSERT_DOUBLE_EQ(res_j[2].v[5], result.df_dp[8]);
}

TEST(SolverUtils, PerformanceDoubleTest) { 
    const double aa[3] = {M_PI / 6, 0, 0};
    Eigen::Vector3d pt = {1, 1, 1};
    
    int count = 1000000;
    std::vector<Eigen::Vector3d> pts(count, pt);

    std::clock_t cpu_start = std::clock();
    for(int i = 0; i < count; ++i) {
        auto res = Rotation<double>::df(aa, pts[i].data());
    }
    
    std::clock_t cpu_end = std::clock();
    float cpu_duration = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    //std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;
}

TEST(SolverUtils, PerformanceVectorTest) { 
    const double aa[3] = {M_PI / 6, 0, 0};
    Eigen::Vector3d pt = {1, 1, 1};
    
    int count = 1000000;
    std::vector<Eigen::Vector3d> pts(count, pt);
    
    std::clock_t cpu_start = std::clock();
    
    auto res = Rotation<double>::df(aa, pts);
    
    std::clock_t cpu_end = std::clock();
    float cpu_duration = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    //std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;
}

TEST(SolverUtils, TransformationTest) { 

    const double pose[6] = {M_PI / 5, 0, 0, 1, 2, 3};
    const double pt[3] = {1, 1, 1};
    
    auto res = Transformation<double>::df(pose, pt);
    //std::cout << res << std::endl << std::endl;

    // using ceres::Jet for comparosing 
    using JetT = ceres::Jet<double, 9>;
    Eigen::Vector<JetT, 6> pose_j;
    Eigen::Vector3<JetT> pt_j;
    for(int i = 0; i < 6; ++i)  {
        pose_j[i] = JetT(pose[i], i);
    }
    for(int i = 0; i < 3; ++i)  {
        pt_j[i] = JetT(pt[i], i+6);
    }

    Eigen::Vector3<JetT> res_j;
    Transformation<JetT>::f(pose_j.data(), pt_j.data(), res_j.data());
    //std::cout << res_j << std::endl << std::endl;
}

TEST(SolverUtils, TransformationPoseOnlyTest) { 

    const double pose[6] = {M_PI / 5, 0, 0, 1, 2, 3};
    const double pt[3] = {1, 1, 1};

    auto res = Transformation<double>::df_dps(pose, pt);
    //std::cout << res << std::endl << std::endl;

    // using ceres::Jet for comparosing 
    using JetT = ceres::Jet<double, 9>;
    Eigen::Vector<JetT, 6> pose_j;
    Eigen::Vector3<JetT> pt_j;
    for(int i = 0; i < 6; ++i)  {
        pose_j[i] = JetT(pose[i], i);
    }
    for(int i = 0; i < 3; ++i)  {
        pt_j[i] = JetT(pt[i]);
    }

    Eigen::Vector3<JetT> res_j;
    Transformation<JetT>::f(pose_j.data(), pt_j.data(), res_j.data());

    //std::cout << res_j << std::endl << std::endl;
}

TEST(SolverUtils, ProjectionTest) { 

    const double pose[6] = {M_PI / 5, 0, 0, 1, 2, 3};
    const double pt[3] = {1, 1, 1};
    
    auto res = PerspectiveProjection<double>::df_dps(pose, pt);
    //std::cout << res << std::endl << std::endl;

    // using ceres::Jet for comparosing 
    using JetT = ceres::Jet<double, 9>;
    Eigen::Vector<JetT, 6> pose_j;
    Eigen::Vector3<JetT> pt_j;
    for(int i = 0; i < 6; ++i)  {
        pose_j[i] = JetT(pose[i], i);
    }
    for(int i = 0; i < 3; ++i)  {
        pt_j[i] = JetT(pt[i], i+6);
    }

    Eigen::Vector2<JetT> res_j;
    PerspectiveProjection<JetT>::f(pose_j.data(), pt_j.data(), res_j.data());

    //std::cout << res_j << std::endl << std::endl;
}

TEST(TriangulationTest, TriangulatePointDepthsTest) {
    // world 3d point
    const Eigen::Vector3d x(1, 1, 5);
    const Eigen::Vector3d err(0, 0, 0);
    // second frame transform
    Eigen::Isometry3d T;
    T.setIdentity();
    T.linear() = Eigen::AngleAxisd(0.0, Eigen::Vector3d(1,1,1).normalized()).toRotationMatrix();
    T.translation() = Eigen::Vector3d(2, 0, 0);

    const Eigen::Vector3d x0 = x - err;
    const Eigen::Vector3d x1 = T.inverse(Eigen::Isometry) * (x + err);
    
    const Eigen::Vector2d y0(x0[0] / x0[2], x0[1] / x0[2]);
    const Eigen::Vector2d y1(x1[0] / x1[2], x1[1] / x1[2]);

    double d1, d2;
    TriangulatePointDepths(y0, T, y1, d1, d2);

    ASSERT_EQ(d1, 5);
    ASSERT_EQ(d2, 5);
}

TEST(SolverUtils, UnitQuaternionTest) { 
    const double aa[3] = {M_PI / 6, 0.1, 0.2};
    const double pt[3] = {1, 2, 1};

    Eigen::Matrix3d rot;
    ceres::AngleAxisToRotationMatrix(aa, rot.data());   
    Eigen::Quaterniond qt(rot);
    Eigen::Vector4d qt_vec = qt.coeffs();
    Eigen::Vector3d res_f;

    Eigen::Matrix<double, 3, 4> J;
    RotationUnitQuaternion<double>::df_dq(qt_vec.data(), pt, res_f.data(), J.data());
//    std::cout << res_f.transpose() << std::endl;
//    std::cout << J << std::endl;

    using JetT = ceres::Jet<double, 4>;
    Eigen::Vector4<JetT> qt_j;
    Eigen::Vector3<JetT> pt_j;

    for(int i = 0; i < 4; ++i)  {
        qt_j[i] = JetT(qt_vec[i], i);
    }

    for(int i = 0; i < 3; ++i)  {
        pt_j[i] = JetT(pt[i]);
    }

    Eigen::Vector3<JetT> res_j;
    RotationUnitQuaternion<JetT>::f(qt_j.data(), pt_j.data(), res_j.data());
//    std::cout << res_j << std::endl << std::endl;
}

TEST(SolverUtils, PnPTest) { 
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::Vector<double, 6> pose = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(pose.data(), R.data());
    Eigen::Vector3d t(pose.data() + 3);

    size_t size = 1000;
    std::vector<Eigen::Vector3d> map(size);
    std::vector<Eigen::Vector2d> frame_t(size);

    for(size_t i = 0; i < size; ++i) {
        Eigen::Vector3d point_3d(dist(gen), dist(gen), dist(gen) + 2);
        Eigen::Vector3d point_3d_t = R * point_3d + t;
        map[i] = point_3d;
        frame_t[i] = Eigen::Vector2d(point_3d_t[0]/point_3d_t[2], point_3d_t[1]/point_3d_t[2]);
    }    

    Eigen::Vector<double, 6> slvd_pose;
    slvd_pose.setZero();
    
    std::cout << slvd_pose.transpose() << std::endl;
    PnPSolver::Cofiguration pnp_config;
    
    std::clock_t cpu_start = std::clock();
    PnPSolver::SolvePose(map, frame_t, slvd_pose, pnp_config);
    std::clock_t cpu_end = std::clock();
    float cpu_duration = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;
    std::cout << slvd_pose.transpose() << std::endl;
}

TEST(SolverUtils, HomographyTest) {
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::Isometry3d gt_T;
    gt_T.setIdentity();
    gt_T.translation() = Eigen::Vector3d(0.1, 0.1, 0.1);
    gt_T.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d(1, 1, 1).normalized()).toRotationMatrix();

    size_t size = 1000;
    std::vector<Eigen::Vector3d> map(size);
    std::vector<Eigen::Vector2d> frame_w(size);
    std::vector<Eigen::Vector2d> frame_t(size);

    for(size_t i = 0; i < size; ++i) {
        Eigen::Vector3d point_3d(dist(gen), dist(gen), dist(gen) + 2);
        Eigen::Vector3d point_3d_t = gt_T.inverse() * point_3d;
        map[i] = point_3d;
        frame_w[i] = Eigen::Vector2d(point_3d[0]/point_3d[2], point_3d[1]/point_3d[2]);
        frame_t[i] = Eigen::Vector2d(point_3d_t[0]/point_3d_t[2], point_3d_t[1]/point_3d_t[2]);

        double d0, d1;
        TriangulatePointDepths(frame_w[i], gt_T, frame_t[i], d0, d1);
        ASSERT_FLOAT_EQ(d0, point_3d[2]);
        ASSERT_FLOAT_EQ(d1, point_3d_t[2]);
    }  

    HomographySolver::Cofiguration h_config;
    std::clock_t cpu_start = std::clock();
    Eigen::Isometry3d hmg_T;
    HomographySolver::Solve(frame_w, frame_t, hmg_T, h_config);
    std::clock_t cpu_end = std::clock();
    float cpu_duration = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;

    Eigen::Matrix3d R(hmg_T.rotation());
    Eigen::Vector3d aa;
    ceres::RotationMatrixToAngleAxis(R.data(), aa.data());
    std::cout << "aa - " << aa.transpose() << std::endl;
    std::cout << "t - " << hmg_T.translation().transpose() << std::endl;

}

TEST(SolverUtils, HomographyCeresQTest) {
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::Vector3d aa(0.1, 0.1, 0.1);
    Eigen::Vector3d t(0.1, 0.1, 0.1);


    Eigen::Isometry3d gt_T;
    gt_T.setIdentity();
    gt_T.translation() = t;
    gt_T.linear() = Eigen::AngleAxisd(aa.norm(), aa.normalized()).toRotationMatrix();

    size_t size = 1000;
    std::vector<Eigen::Vector3d> map(size);
    std::vector<Eigen::Vector2d> frame_w(size);
    std::vector<Eigen::Vector2d> frame_t(size);

    for(size_t i = 0; i < size; ++i) {
        Eigen::Vector3d point_3d(dist(gen), dist(gen), dist(gen) + 2);
        Eigen::Vector3d point_3d_t = gt_T.inverse() * point_3d;
        map[i] = point_3d;
        frame_w[i] = Eigen::Vector2d(point_3d[0]/point_3d[2], point_3d[1]/point_3d[2]);
        frame_t[i] = Eigen::Vector2d(point_3d_t[0]/point_3d_t[2], point_3d_t[1]/point_3d_t[2]);

        double d0, d1;
        TriangulatePointDepths(frame_w[i], gt_T, frame_t[i], d0, d1);
        ASSERT_FLOAT_EQ(d0, point_3d[2]);
        ASSERT_FLOAT_EQ(d1, point_3d_t[2]);
    }  

    HomographySolver::Cofiguration h_config;
    std::clock_t cpu_start = std::clock();
    Eigen::Vector3d aa_ = Eigen::Vector3d::Zero();   
    
    Eigen::Vector4d q_;
    ceres::AngleAxisToQuaternion(aa_.data(), q_.data());   
    Eigen::Vector3d t_(1, 0, 0);
    HomographySolver::Solve_Ceres_qt(frame_w, frame_t, q_, t_);
    
    std::clock_t cpu_end = std::clock();
    float cpu_duration = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;

    ceres::QuaternionToAngleAxis(q_.data(), aa_.data());

    std::cout << "aa - " << aa_.transpose() << std::endl;
    std::cout << "t - " << t_.transpose() << std::endl;
}

TEST(SolverUtils, HomographyCeresAATest) {
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::Vector3d aa(0.1, 0.1, 0.1);
    Eigen::Vector3d t(0.1, 0.1, 0.1);


    Eigen::Isometry3d gt_T;
    gt_T.setIdentity();
    gt_T.translation() = t;
    gt_T.linear() = Eigen::AngleAxisd(aa.norm(), aa.normalized()).toRotationMatrix();

    size_t size = 1000;
    std::vector<Eigen::Vector3d> map(size);
    std::vector<Eigen::Vector2d> frame_w(size);
    std::vector<Eigen::Vector2d> frame_t(size);

    for(size_t i = 0; i < size; ++i) {
        Eigen::Vector3d point_3d(dist(gen), dist(gen), dist(gen) + 2);
        Eigen::Vector3d point_3d_t = gt_T.inverse() * point_3d;
        map[i] = point_3d;
        frame_w[i] = Eigen::Vector2d(point_3d[0]/point_3d[2], point_3d[1]/point_3d[2]);
        frame_t[i] = Eigen::Vector2d(point_3d_t[0]/point_3d_t[2], point_3d_t[1]/point_3d_t[2]);

        double d0, d1;
        TriangulatePointDepths(frame_w[i], gt_T, frame_t[i], d0, d1);
        ASSERT_FLOAT_EQ(d0, point_3d[2]);
        ASSERT_FLOAT_EQ(d1, point_3d_t[2]);
    }  

    HomographySolver::Cofiguration h_config;
    std::clock_t cpu_start = std::clock();
    Eigen::Vector3d aa_ = Eigen::Vector3d::Zero();   
    
    Eigen::Vector3d t_(1, 0, 0);
    HomographySolver::Solve_Ceres(frame_w, frame_t, aa_, t_);
    
    std::clock_t cpu_end = std::clock();
    float cpu_duration = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;

    std::cout << "aa - " << aa_.transpose() << std::endl;
    std::cout << "t - " << t_.transpose() << std::endl;
}

TEST(Utils, PoseConvertionTest) {
    Eigen::Vector<double, 6> pose_se(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
    Eigen::Isometry3d pose_is = Convert(pose_se);
    Eigen::Vector<double, 6> pose_se_ = Convert(pose_is);

    for(size_t i = 0; i < 6; ++i) {
        ASSERT_DOUBLE_EQ(pose_se[i], pose_se_[i]);
    } 
}

TEST(SolverUtils, RotationZeroTest) { 
    const double aa[3] = {0, 0, 0};
    const double pt[3] = {1, 2, 3};
    
    auto result = Rotation<double>::df(aa, pt);

    // using ceres::Jet for comparosing 
    using JetT = ceres::Jet<double, 6>;
    Eigen::Vector3<JetT> pt_j, aa_j;
    for(int i = 0; i < 3; ++i)  {
        aa_j[i] = JetT(aa[i], i);
        pt_j[i] = JetT(pt[i], i+3);
    }

    Eigen::Vector3<JetT> res_j;
    Rotation<JetT>::f(aa_j.data(), pt_j.data(), res_j.data());

    ASSERT_DOUBLE_EQ(res_j[0].a, result.Rpt[0]);
    ASSERT_DOUBLE_EQ(res_j[1].a, result.Rpt[1]);
    ASSERT_DOUBLE_EQ(res_j[2].a, result.Rpt[2]);

    ASSERT_DOUBLE_EQ(res_j[0].v[0], result.df_daa[0]);
    ASSERT_DOUBLE_EQ(res_j[1].v[0], result.df_daa[1]);
    ASSERT_DOUBLE_EQ(res_j[2].v[0], result.df_daa[2]);

    ASSERT_DOUBLE_EQ(res_j[0].v[1], result.df_daa[3]);
    ASSERT_DOUBLE_EQ(res_j[1].v[1], result.df_daa[4]);
    ASSERT_DOUBLE_EQ(res_j[2].v[1], result.df_daa[5]);

    ASSERT_DOUBLE_EQ(res_j[0].v[2], result.df_daa[6]);
    ASSERT_DOUBLE_EQ(res_j[1].v[2], result.df_daa[7]);
    ASSERT_DOUBLE_EQ(res_j[2].v[2], result.df_daa[8]);

    ASSERT_DOUBLE_EQ(res_j[0].v[3], result.df_dp[0]);
    ASSERT_DOUBLE_EQ(res_j[1].v[3], result.df_dp[1]);
    ASSERT_DOUBLE_EQ(res_j[2].v[3], result.df_dp[2]);

    ASSERT_DOUBLE_EQ(res_j[0].v[4], result.df_dp[3]);
    ASSERT_DOUBLE_EQ(res_j[1].v[4], result.df_dp[4]);
    ASSERT_DOUBLE_EQ(res_j[2].v[4], result.df_dp[5]);

    ASSERT_DOUBLE_EQ(res_j[0].v[5], result.df_dp[6]);
    ASSERT_DOUBLE_EQ(res_j[1].v[5], result.df_dp[7]);
    ASSERT_DOUBLE_EQ(res_j[2].v[5], result.df_dp[8]);
}

TEST(SolverUtils, TransformationZeroPoseOnlyTest) { 

    const double pose[6] = {0, 0, 0, 0, 0, 0};
    const double pt[3] = {1, 2, 3};

    auto res = Transformation<double>::df_dps(pose, pt);
    //std::cout << res << std::endl << std::endl;

    auto res_zero = Transformation<double>::df_dps_zero(pt);
    //std::cout << res_zero << std::endl << std::endl;

    // using ceres::Jet for comparosing 
    using JetT = ceres::Jet<double, 9>;
    Eigen::Vector<JetT, 6> pose_j;
    Eigen::Vector3<JetT> pt_j;
    for(int i = 0; i < 6; ++i)  {
        pose_j[i] = JetT(pose[i], i);
    }
    for(int i = 0; i < 3; ++i)  {
        pt_j[i] = JetT(pt[i]);
    }

    Eigen::Vector3<JetT> res_j;
    Transformation<JetT>::f(pose_j.data(), pt_j.data(), res_j.data());

//    std::cout << res_j << std::endl << std::endl;
}

TEST(SolverUtils, PerspectivePointJTest) { 
    
    const double f = 450;
    const Eigen::Vector3d pnt(1, 2, 3);
    auto prj = PerspectiveProjection<double>::f(f, pnt);
    auto dprj_dpnt = PerspectiveProjection<double>::df_dpnt(f, pnt);
    
    //std::cout << prj.transpose() << std::endl;
    //std::cout << dprj_dpnt << std::endl;
     
    using JetT = ceres::Jet<double, 3>;
    Eigen::Vector3<JetT> pnt_j;
    for(int i = 0; i < 3; ++i)  {
        pnt_j[i] = JetT(pnt[i], i);
    }
    JetT f_j(f);

    auto prj_j = PerspectiveProjection<JetT>::f(f_j, pnt_j);

    //std::cout << prj_j << std::endl;
}

TEST(SolverUtils, PerspectiveZeroPoseJTest) { 
    
    const double f = 1;
    const Eigen::Vector3d pnt(1, 2, 3);
    auto prj = PerspectiveProjection<double>::f(f, pnt);
    auto dprj_dps_zero = PerspectiveProjection<double>::df_dps_zero(f, pnt);
    
    std::cout << prj.transpose() << std::endl;
    std::cout << dprj_dps_zero << std::endl;

    Eigen::Vector<double, 6> pose(0, 0, 0, 0, 0, 0); 
    using JetT = ceres::Jet<double, 6>;
    
    Eigen::Vector<JetT, 6> pose_j;
    for(int i = 0; i < 6; ++i)  {
        pose_j[i] = JetT(pose[i], i);
    }

    Eigen::Vector3<JetT> pnt_j;
    for(int i = 0; i < 3; ++i)  {
        pnt_j[i] = JetT(pnt[i]);
    }

    JetT f_j(f);

    auto prj_j = PerspectiveProjection<JetT>::f(pose_j, f_j, pnt_j);

    std::cout << prj_j << std::endl;
}

TEST(SolverUtils, PoseToEigenTest) { 
    Eigen::Vector3d pnt = {1, 2, 3}; 
    Eigen::Vector<double, 6> pose = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    Eigen::Vector3d trans_pnt = Transformation<double>::f(pose, pnt);

    Eigen::Isometry3d ism = Transformation<double>::Convert(pose.data());
    Eigen::Vector3d trans_pnt_ = ism * pnt;

    ASSERT_DOUBLE_EQ(trans_pnt[0], trans_pnt_[0]);
    ASSERT_DOUBLE_EQ(trans_pnt[1], trans_pnt_[1]);
    ASSERT_DOUBLE_EQ(trans_pnt[2], trans_pnt_[2]);

    Eigen::Vector<double, 6> pose_ = Transformation<double>::Convert(ism);

    //std::cout << pose_.transpose() << std::endl;
}

auto ThreadTestFunc(int a, int b, bool sleep) {
    std::cout << "ThreadTestFunc(" << a << "," << b << ")" << std::endl;
    if (sleep) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "Done." << std::endl;

}

TEST(ThreadUtils, ThreadTest) {
    auto t1 = ExecuteInThread(-1, ThreadTestFunc, 1, 2, true);
    auto t2 = ExecuteInThread(1, ThreadTestFunc, 3, 4, false);
    std::cout << "Waiting ... " << std::endl;
    t1->join();
    t2->join();
    std::cout << "Main done. " << std::endl;
}

auto ReadFunction(SPSCQueue<int>& queue) {
    while(queue.Size()) {
        int value;
        if (queue.Read(value)) {
            std::cout << "Read element " << value << std::endl;
        } else {
            std::cout << "Failed to read an element." << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));   
    }
    
    std::cout << "Nothing to read, done." << std::endl;
}

auto WriteAllFunction(SPSCQueue<int>& queue) {
    
    size_t i = 0;
    size_t count = queue.Capacity() * 2;

    while(i < count) {
        if (queue.Write(i)) {
            std::cout << "Write element " << i << std::endl;
            ++i;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    std::cout << "Nothing to write, done." << std::endl;
}

TEST(ThreadUtils, QueueTest) {
    SPSCQueue<int> queue(10);

    auto t1 = ExecuteInThread(-1, WriteAllFunction, queue);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto t2 = ExecuteInThread(-1, ReadFunction, queue);

    t2->join();
    t1->join();
    std::cout << "Test done. " << std::endl;
}

int test_func(int x) {
    if (x <= 0)
        return 0;

    return x + test_func(x-1);
}

TEST(ThreadUtils, ThreadPoolTest) {
    ThreadPool thread_pool;

    thread_pool.Start();
    
    for(int i = 0; i < 100; ++i) {
        thread_pool.QueueJob([&i]{
            test_func(100);
        });
    }

    while(thread_pool.Busy()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    thread_pool.Stop();
}

struct TestStruct {
    int v_[2];
};

TEST(ThreadUtils, MemoryPoolTest) {
    int size = 10; 
    MemoryPool<TestStruct> m_pool(size);

    std::vector<TestStruct*> data(size);

    ASSERT_EQ(size, m_pool.Capasity());
    ASSERT_EQ(0, m_pool.Size());

    for(int i = 0; i < size; ++i) {
        data[i] = m_pool.allocate(TestStruct{i, i + 1});
    }

    ASSERT_EQ(size, m_pool.Size());

    for(int i = 0; i < size; ++i) {
        m_pool.deallocate(data[i]);
    }

        ASSERT_EQ(0, m_pool.Size());

}

void Node_1_func(QueueWithStatus<int>* in_queue, SPSCQueue<int>* out_queue) 
{
    int i = 0;
    while(i < 20) {
        // computation

        if (out_queue->Write(i)) {
            ++i;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } 
    }
}

void Node_2_func(QueueWithStatus<int>* in_queue, SPSCQueue<int>* out_queue, int multiplyer) 
{
    while(true) {
        // TODO: add abort mechanism

        int in_value;
        if (in_queue->queue_->Read(in_value)) {
            
            // computation
            const int out_value = in_value * multiplyer;

            while(!out_queue->Write(out_value)) {
                // TODO: add abort mechanism
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

        } else {
            // check input node status and exit if input node done.
            if (NodeStatus::done == in_queue->status_->load()) {
                return;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void Node_3_func(QueueWithStatus<int>* in_queue, SPSCQueue<int>* out_queue) 
{
    while(true) {
        // TODO: add abort mechanism
        
        int in_value;
        if (in_queue->queue_->Read(in_value)) {
            
            std::cout << "Multiplication result: " << in_value << std::endl;
        } else {
            // check input node status and exit if input node done.
            if (NodeStatus::done == in_queue->status_->load()) {
                return;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}


TEST(ThreadUtils, PipelineTest) {
    PipelineNode<int, int> node_1(nullptr, 10);
    PipelineNode<int, int> node_2(node_1.GetOutQueue(), 10);
    PipelineNode<int, int> node_3(node_2.GetOutQueue(), 10);

    node_1.Start(Node_1_func);
    node_2.Start(Node_2_func, 2);
    node_3.Start(Node_3_func);

    node_1.Join();
    node_2.Join();
    node_3.Join();
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

