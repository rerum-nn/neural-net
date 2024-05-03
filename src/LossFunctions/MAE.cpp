#include "MAE.h"

namespace neural_net {

double MAE::Loss(const Matrix& present, const Matrix& expected) const {
    return (expected - present).cwiseAbs().sum() / expected.size();
}

Matrix MAE::LossGradient(const Matrix& present, const Matrix& expected) const {
    Matrix diff = expected - present;
    return -diff.cwiseQuotient(diff.cwiseAbs()) / present.size();
}

}  // namespace neural_net
