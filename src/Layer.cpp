#include "Layer.h"

#include <EigenRand/EigenRand>
#include <cassert>

namespace neural_net {

Layer::Layer(size_t input, size_t output, ActivationFunction func)
    : weights_(output, input), bias_(output), activation_func_(std::move(func)) {
}
void Layer::NormalRandomInit(int seed) {
    Eigen::Rand::P8_mt19937_64 generator(seed);
    weights_ = Eigen::Rand::normalLike(weights_, generator);
    bias_ = Eigen::Rand::normalLike(bias_, generator);
}
void Layer::Forward(Vector* input_vector) const {
    assert(input_vector && "input_vector in Layer Forward method can`t be nullptr");
    *input_vector = weights_ * *input_vector + bias_;
    activation_func_->Apply(input_vector);
}

}  // namespace neural_net
