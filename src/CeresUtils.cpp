#include "utils/CeresUtils.h"
#include <thread>

bool Optimize(
        ceres::Problem& problem,
        bool verbal,
        int max_iterations,
        double threshold)
{
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;

    options.minimizer_progress_to_stdout = verbal;
    options.max_num_iterations = max_iterations;
    options.function_tolerance = threshold;
    options.gradient_tolerance = 1e-4 * threshold;
    options.parameter_tolerance = 1e-2 * threshold;

    options.num_threads = std::thread::hardware_concurrency();
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    ceres::Solve(options, &problem, &summary);

    if(verbal) {
        std::cerr << summary.BriefReport() << std::endl; 
        std::cerr << summary.message << std::endl; 
    }

    return ceres::CONVERGENCE == summary.termination_type && summary.num_successful_steps > 1;
}
