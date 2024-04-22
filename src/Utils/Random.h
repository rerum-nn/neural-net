#pragma once

#include "EigenRand/EigenRand"
#include "Types.h"

#include <random>

namespace neural_net {

class Random {
public:
    static Random& Instance(int seed = 1337);

    static Matrix Normal(Index rows, Index cols);
    static PermutationMatrix Permutation(Index size);

private:
    Random(int seed);

    Matrix NormalImpl(Index rows, Index cols);
    PermutationMatrix PermutationImpl(Index size);

    Eigen::Rand::P8_mt19937_64 generator_;
};

}  // namespace neural_net
