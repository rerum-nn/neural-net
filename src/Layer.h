#pragma once

#include <functional>
#include <memory>

#include "ActivationFunctions/ActivationFunction.h"
#include "LinearPrimitives.h"
#include "WeightGenerators/NormalRandom.h"
#include "WeightGenerators/RandomGenerator.h"

namespace neural_net {
class Layer {
public:
    Layer(size_t input, size_t output, ActivationFunction func,
          RandomGenerator&& generator = NormalRandom());

    DataVector Forward(const DataVector& input_vector) const;
    void Forward(DataVector* input_vector) const;

private:
    WeightMatrix weights_;
    DataVector bias_;

    ActivationFunction activation_func_;
};

}  // namespace neural_net
