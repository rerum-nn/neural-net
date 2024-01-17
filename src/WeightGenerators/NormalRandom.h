#pragma once

#include "RandomGenerator.h"

namespace neural_net {

class NormalRandom {
public:
    NormalRandom(double mean = 0, double stddev = 1, unsigned long seed = std::random_device()());

    double Next();

private:
    std::default_random_engine dre_;
    std::normal_distribution<> nd_;
};

}  // namespace neural_net

