#pragma once

#include "../LinearPrimitives.h"

namespace neural_net {

class Sigmoid {
public:
    void Apply(DataVector* data_vector) const;
    WeightMatrix Derivative(const DataVector& values) const;
};

}  // namespace neural_net
