#pragma once

#include <string>
#include <chrono>

namespace neural_net {

class Timer {
public:
    Timer();

    void Reset();
    double Elapsed() const;
    std::string GetTimerString() const;

private:
    std::chrono::time_point<std::chrono::system_clock> start_;
};

}  // namespace neural_net
