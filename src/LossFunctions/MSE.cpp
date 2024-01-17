#include "MSE.h"

#include <algorithm>
#include <cassert>

namespace neural_net {
double MSE::Loss(const DataVector& present, const DataVector& expected) {
    assert(present.size() == expected.size() && "present and expected sizes must be the same");
    double res = 0;
    auto present_it = present.begin();
    auto expected_it = expected.begin();
    while (present_it != present.end()) {
        res += (*expected_it - *present_it) * (*expected_it - *present_it);
    }

    return res / present.size();
}
DataVector MSE::LossGradient(const DataVector& present, const DataVector& expected) {
    assert(present.size() == expected.size() && "present and expected sizes must be the same");
    DataVector nabla(present.size());

    std::transform(present.begin(), present.end(), expected.begin(), nabla.begin(),
                   [](double d1, double d2) { return d2 - d1; });

    return nabla;
}
}  // namespace neural_net
