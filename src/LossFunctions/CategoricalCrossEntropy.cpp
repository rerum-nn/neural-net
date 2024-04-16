#include "CategoricalCrossEntropy.h"

namespace neural_net {
double CategoricalCrossEntropy::Loss(const Vector& present, const Vector& expected) const {
    return -(expected.array() * present.array().log()).sum();
}

RowVector CategoricalCrossEntropy::LossGradient(const Vector& present,
                                                const Vector& expected) const {
    return -expected.array() / present.array();
}
}  // namespace neural_net