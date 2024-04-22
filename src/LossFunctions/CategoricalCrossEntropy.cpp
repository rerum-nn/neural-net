#include "CategoricalCrossEntropy.h"

namespace neural_net {

double CategoricalCrossEntropy::Loss(const Matrix& present, const Matrix& expected) const {
    return -(expected.array() * (present.array() + kEpsilon).log()).sum() / present.rows();
}

Matrix CategoricalCrossEntropy::LossGradient(const Matrix& present, const Matrix& expected) const {
    return -expected.array() / (present.array() + kEpsilon) / present.rows();
}

}  // namespace neural_net
