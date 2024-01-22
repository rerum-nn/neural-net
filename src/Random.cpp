#include "Random.h"

namespace neural_net {

Random::Random(int seed) : generator_(seed) {
}

Random& Random::Instance(int seed) {
    static Random instance(seed);
    return instance;
}

Matrix Random::Normal(Index rows, Index columns) {
    return Instance().NormalImpl(rows, columns);
}

Matrix Random::NormalImpl(Index rows, Index columns) {
    return Eigen::Rand::normal<Matrix>(rows, columns, generator_);
}

}  // namespace neural_net
