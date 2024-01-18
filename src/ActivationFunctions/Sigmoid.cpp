#include "Sigmoid.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace neural_net {

void Sigmoid::Apply(Vector* data_vector) const {
    assert(data_vector && "data_vector can't be nullptr");
    std::for_each(data_vector->begin(), data_vector->end(),
                  [](double& d) { d = 1. / (1. + std::exp(-d)); });
}
Matrix Sigmoid::Derivative(const Vector& values) const {
    Matrix jacobian(values.size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        double sigmoid = 1. / (1. + std::exp(-values[i]));
        jacobian(i, i) = sigmoid * (1 - sigmoid);
    }
    return jacobian;
}
}  // namespace neural_net
