#pragma once

#include "ActivationFunctions/ActivationFunction.h"
#include "Layer.h"
#include "LossFunctions/LossFunction.h"
#include "Types.h"

#include <vector>

namespace neural_net {

class Network {
public:
    Network(std::initializer_list<size_t> layer_sizes,
            std::initializer_list<ActivationFunction> functions);

    Vector Predict(const Vector& input_data);

private:
    std::vector<Layer> layers_;
};

}  // namespace neural_net
