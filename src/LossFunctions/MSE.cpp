#include "MSE.h"

#include <algorithm>
#include <cassert>

namespace neural_net {

double MSE::Loss(const Vector& present, const Vector& expected) const {
    assert(present.size() == expected.size() && "present and expected sizes must be the same");
    double norm = (expected - present).norm();
    return norm * norm;
}

RowVector MSE::LossGradient(const Vector& present, const Vector& expected) const {
    assert(present.size() == expected.size() && "present and expected sizes must be the same");
    return (2 * (expected - present)).transpose();
}

}  // namespace neural_net
