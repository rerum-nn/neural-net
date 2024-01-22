#include "MSE.h"

#include <algorithm>
#include <cassert>

namespace neural_net {

double MSE::Loss(const Vector& present, const Vector& expected) {
    assert(present.size() == expected.size() && "present and expected sizes must be the same");
    return (expected - present).norm() / present.size();
}

Vector MSE::LossGradient(const Vector& present, const Vector& expected) {
    assert(present.size() == expected.size() && "present and expected sizes must be the same");
    return 2 * (expected - present);
}

}  // namespace neural_net
