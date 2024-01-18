#pragma once

#include <functional>
#include <memory>
#include <random>

#include "ActivationFunctions/ActivationFunction.h"
#include "LinearPrimitives.h"

namespace neural_net {
class Layer {
public:
    Layer(size_t input, size_t output, ActivationFunction func);

    void NormalRandomInit(int seed = std::random_device()());
    void Forward(Vector* input_vector) const;

private:
    Matrix weights_;
    Vector bias_;

    ActivationFunction activation_func_;
};

}  // namespace neural_net
