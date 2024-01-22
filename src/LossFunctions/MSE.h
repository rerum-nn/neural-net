#pragma once

#include "../Types.h"

namespace neural_net {

class MSE {
public:
    static double Loss(const Vector& present, const Vector& expected);
    static Vector LossGradient(const Vector& present, const Vector& expected);
};

}  // namespace neural_net
