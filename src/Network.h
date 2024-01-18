#pragma once

#include <memory>
#include <vector>

#include "ActivationFunctions/ActivationFunction.h"
#include "Layer.h"
#include "LinearPrimitives.h"
#include "LossFunctions/LossFunction.h"

namespace neural_net {

class Network {
public:
    Network(std::initializer_list<size_t> layer_sizes,
            std::initializer_list<ActivationFunction> functions);

    Vector Predict(const Vector& input_data);
    Vector Predict(const std::initializer_list<double>& input_data);

private:
    std::vector<Layer> layers_;
};

}  // namespace neural_net
