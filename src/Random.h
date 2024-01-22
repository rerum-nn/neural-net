#pragma once

#include "Types.h"

#include <EigenRand/EigenRand>

#include <random>

namespace neural_net {

class Random {
public:
    static Random& Instance(int seed = std::random_device()());

    static Matrix Normal(Index rows, Index columns);

private:
    Random(int seed);

    Matrix NormalImpl(Index rows, Index columns);

    Eigen::Rand::P8_mt19937_64 generator_;
};

}  // namespace neural_net
