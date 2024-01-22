#include "Layer.h"

#include "Random.h"

namespace neural_net {

Layer::Layer(Index input, Index output, ActivationFunction func)
    : weights_(Random::Normal(output, input)),
      bias_(Random::Normal(output, 1)),
      activation_func_(std::move(func)) {
}

Vector Layer::Forward(const Vector& input_vector) const {
    return activation_func_->Apply(weights_ * input_vector + bias_);
}

}  // namespace neural_net
