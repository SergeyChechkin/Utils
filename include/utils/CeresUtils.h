#pragma once

#include <ceres/ceres.h>

bool Optimize(
        ceres::Problem& problem,
        bool verbal = true,
        int max_iterations = 100,
        double threshold = 1e-6);
