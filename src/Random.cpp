#include "Random.h"

namespace neural_net {

Random::Random(int seed) : generator_(seed) {
}

Random& Random::Instance(int seed) {
    static Random instance(seed);
    return instance;
}

Matrix Random::Normal(Index rows, Index cols) {
    return Instance().NormalImpl(rows, cols);
}

Matrix Random::NormalImpl(Index rows, Index cols) {
    return Eigen::Rand::normal<Matrix>(rows, cols, generator_);
}

}  // namespace neural_net
