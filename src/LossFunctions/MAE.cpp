#include "MAE.h"

namespace neural_net {

float MAE::Loss(const Matrix& present, const Matrix& expected) const {
    assert(present.cols() == expected.cols() && present.rows() == expected.rows());
    return (expected - present).cwiseAbs().sum() / expected.size();
}

Matrix MAE::LossGradient(const Matrix& present, const Matrix& expected) const {
    assert(present.cols() == expected.cols() && present.rows() == expected.rows());
    Matrix diff = expected - present;
    return -diff.cwiseQuotient(diff.cwiseAbs()) / present.size();
}

}  // namespace neural_net
