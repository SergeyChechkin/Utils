/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#include "utils/solver/Rotation.h"
#include "utils/solver/Transformation.h"
#include <ceres/jet.h>
#include <gtest/gtest.h>

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
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;
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
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;
}

TEST(SolverUtils, TransformationTest) { 

    const double pose[6] = {M_PI / 5, 0, 0, 1, 2, 3};
    const double pt[3] = {1, 1, 1};
    
    //Eigen::Vector3d res_d;
    //Transformation<double>::f(pose, pt, res_d.data());
    //std::cout << res_d << std::endl;

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

    auto res = Transformation<double>::df_dp(pose, pt);
    std::cout << res << std::endl << std::endl;

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

    std::cout << res_j << std::endl << std::endl;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}