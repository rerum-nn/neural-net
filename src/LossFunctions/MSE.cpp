#include "MSE.h"

#include <cassert>

namespace neural_net {

double MSE::Loss(const Matrix& present, const Matrix& expected) const {
    assert(present.cols() == expected.cols() && present.rows() == expected.rows());
    double norm = (expected - present).norm();
    return norm * norm / present.size();
}

Matrix MSE::LossGradient(const Matrix& present, const Matrix& expected) const {
    assert(present.cols() == expected.cols() && present.rows() == expected.rows());
    return 2 * (expected - present) / present.size();
}

}  // namespace neural_net
