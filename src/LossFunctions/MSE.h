#pragma once

#include "../Types.h"

namespace neural_net {

class MSE {
public:
    double Loss(const Vector& present, const Vector& expected);
    Vector LossGradient(const Vector& present, const Vector& expected);
};

}  // namespace neural_net
