#include "Linear.h"

#include "../Random.h"

namespace neural_net {

Linear::Linear(Index input, Index output)
    : weights_(Random::Normal(output, input)),
      bias_(Random::Normal(output, 1)) {
    assert(input > 0 && "input size of layer should be positive");
    assert(output > 0 && "output size of layer should be positive");
}

Vector Linear::Apply(const Vector& input_vector) {
    assert(input_vector.size() == weights_.cols() &&
           "input vector size and layer input size are not consisted");
    input_vector_ = input_vector;
    return weights_ * input_vector + bias_;
}

RowVector Linear::Fit(const RowVector& loss) {
    assert(input_vector_.size() > 0 &&
           "input_vector for fit information hasn't been transferred");
    Matrix weights_delta = (input_vector_ * loss).transpose();
    Vector bias_delta = loss.transpose();
    RowVector next_lose = loss * weights_;
    weights_ -= weights_delta;
    bias_ -= bias_delta;
    return next_lose;
}

}  // namespace neural_net
