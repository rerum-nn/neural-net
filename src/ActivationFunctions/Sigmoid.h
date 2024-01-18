#pragma once

#include "../LinearPrimitives.h"

namespace neural_net {

class Sigmoid {
public:
    void Apply(Vector* data_vector) const;
    Matrix Derivative(const Vector& values) const;
};

}  // namespace neural_net
