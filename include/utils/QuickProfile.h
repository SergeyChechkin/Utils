#pragma once

#include <chrono>
#include <iostream>

class QuickProfile {
public:
    QuickProfile() {
        Start();
    }

    void Start() {
        w_start_ = std::chrono::high_resolution_clock::now();
        cpu_start_ = std::clock();
    }

    void Stop() {
        auto w_end = std::chrono::high_resolution_clock::now();
        std::clock_t cpu_end = std::clock();
        float cpu_duration = 1000.0 * (cpu_end - cpu_start_) / CLOCKS_PER_SEC;
        auto w_duration = std::chrono::duration_cast<std::chrono::milliseconds>(w_end - w_start_);
        float wall_duration = w_duration.count();

        awg_cpu_duration_ += cpu_duration;
        awg_wall_duration_ += wall_duration;
        ++count_;
    }

    void Print() const {
        std::cout << " Time: CPU - " << awg_cpu_duration_ / count_ << " ms. Wall - " << awg_wall_duration_ / count_ << " ms." << std::endl;
    }
private:
    int count_ = 0;
    float awg_cpu_duration_ = 0;
    float awg_wall_duration_ = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> w_start_;
    std::clock_t cpu_start_;
};