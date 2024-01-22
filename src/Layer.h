#pragma once

#include "ActivationFunctions/ActivationFunction.h"
#include "Types.h"

#include <functional>
#include <random>

namespace neural_net {

class Layer {
public:
    Layer(size_t input, size_t output, ActivationFunction func);

    void NormalRandomInit(int seed = std::random_device()());
    Vector Forward(const Vector& input_vector) const;

private:
    Matrix weights_;
    Vector bias_;
    ActivationFunction activation_func_;

    struct {
        Vector input_vector;
        Vector linear_transformed_vector;
    } fit_information_;
};

}  // namespace neural_net
