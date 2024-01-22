#pragma once

#include "ActivationFunctions/ActivationFunction.h"
#include "Types.h"

#include <functional>
#include <random>

namespace neural_net {

class Layer {
public:
    Layer(Index input, Index output, ActivationFunction func);

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
