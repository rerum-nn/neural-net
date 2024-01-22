#include "Sigmoid.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace neural_net {

Vector Sigmoid::Apply(const Vector& data_vector) const {
    Vector res(data_vector.size());
    std::transform(data_vector.begin(), data_vector.end(), res.begin(),
                   [](double d) { return 1. / (1. + std::exp(-d)); });
    return res;
}

Matrix Sigmoid::Derivative(const Vector& values) const {
    Vector derivative_vector(values.size());
    std::transform(values.begin(), values.end(), derivative_vector.begin(), [](double d) {
        double sigmoid = 1. / (1. + std::exp(-d));
        return sigmoid * (1 - sigmoid);
    });
    return derivative_vector.asDiagonal();
}

}  // namespace neural_net
