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
    Matrix matrix = Eigen::Rand::normal<Matrix>(rows, cols, generator_);
    return matrix / matrix.norm();
}

PermutationMatrix Random::Permutation(Index size) {
    return Instance().PermutationImpl(size);
}

PermutationMatrix Random::PermutationImpl(Index size) {
    PermutationMatrix perm(size);
    perm.setIdentity();
    std::shuffle(perm.indices().begin(), perm.indices().end(), generator_);
    return perm;
}

}  // namespace neural_net
