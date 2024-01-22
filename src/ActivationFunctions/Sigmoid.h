#pragma once

#include "../Types.h"

namespace neural_net {

class Sigmoid {
public:
    Vector Apply(const Vector& data_vector) const;
    Matrix Derivative(const Vector& values) const;
};

}  // namespace neural_net
