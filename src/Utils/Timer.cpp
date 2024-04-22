#include "Timer.h"

namespace neural_net {

Timer::Timer() : start_(std::chrono::system_clock::now()) {
}

void Timer::Reset() {
    start_ = std::chrono::system_clock::now();
}

double Timer::Elapsed() const {
    auto end = std::chrono::system_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
}

std::string Timer::GetTimerString() const {
    auto millis = static_cast<int>(Elapsed());
    int minutes = millis / 60000;
    millis -= minutes * 60000;
    int seconds = millis / 1000;
    millis -= seconds * 1000;

    std::string res;
    if (minutes > 0) {
        res += std::to_string(minutes) + " m ";
    }
    if (seconds > 0 || minutes > 0) {
        res += std::to_string(seconds) + " s ";
    }
    res += std::to_string(millis) + " ms";

    return res;
}

}  // namespace neural_net
