#include "Sigmoid.h"

#include <algorithm>
#include <cmath>

namespace neural_net {

Vector Sigmoid::Apply(const Vector& input_vector) {
    input_vector_ = input_vector;
    Vector res(input_vector.size());
    std::transform(input_vector.begin(), input_vector.end(), res.begin(),
                   [](double d) { return 1. / (1. + std::exp(-d)); });
    return res;
}

RowVector Sigmoid::Fit(const RowVector& loss) {
    Vector derivative_vector(input_vector_.size());
    std::transform(input_vector_.begin(), input_vector_.end(), derivative_vector.begin(),
                   [](double d) {
                       double sigmoid = 1. / (1. + std::exp(-d));
                       return sigmoid * (1 - sigmoid);
                   });
    return loss * derivative_vector;
}

}  // namespace neural_net
