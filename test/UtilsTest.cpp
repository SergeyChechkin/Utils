/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#include "utils/solver/Rotation.h"
#include <ceres/jet.h>
#include <gtest/gtest.h>

TEST(SolverUtils, rotationTest) { 
    const double aa[3] = {M_PI / 6, 0, 0};
    const double pt[3] = {1, 1, 1};
    
    Rotation<double>::RotatedPoint result;
    Rotation<double>::df(aa, pt, result);

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
    const double pt[3] = {1, 1, 1};
    
    Rotation<double>::RotatedPoint result;

    std::clock_t cpu_start = std::clock();
    for(int i = 0; i < 10000000; ++i) {
        Rotation<double>::df(aa, pt, result);
    }
    
    std::clock_t cpu_end = std::clock();
    float cpu_duration = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;
}

TEST(SolverUtils, PerformanceFloatTest) { 
    const float aa[3] = {M_PI / 6, 0, 0};
    const float pt[3] = {1, 1, 1};
    
    Rotation<float>::RotatedPoint result;

    std::clock_t cpu_start = std::clock();
    for(int i = 0; i < 10000000; ++i) {
        Rotation<float>::df(aa, pt, result);
    }
    
    std::clock_t cpu_end = std::clock();
    float cpu_duration = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time - " << cpu_duration << " ms." << std::endl;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}