#include "MSE.h"

#include <cassert>

namespace neural_net {

double MSE::Loss(const Matrix& present, const Matrix& expected) const {
    assert(present.size() == expected.size() && "present and expected sizes must be the same");
    double norm = (expected - present).norm();
    return norm * norm / present.rows();
}

Matrix MSE::LossGradient(const Matrix& present, const Matrix& expected) const {
    assert(present.size() == expected.size() && "present and expected sizes must be the same");
    return 2 * (expected - present) / present.rows();
}

}  // namespace neural_net
