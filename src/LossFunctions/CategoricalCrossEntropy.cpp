#include "CategoricalCrossEntropy.h"

namespace neural_net {

float CategoricalCrossEntropy::Loss(const Matrix& present, const Matrix& expected) const {
    assert(present.cols() == expected.cols() && present.rows() == expected.rows());
    return -(expected.array() * (present.array() + kEpsilon).log()).sum() / present.rows();
}

Matrix CategoricalCrossEntropy::LossGradient(const Matrix& present, const Matrix& expected) const {
    assert(present.cols() == expected.cols() && present.rows() == expected.rows());
    return -expected.array() / (present.array() + kEpsilon) / present.rows();
}

}  // namespace neural_net
