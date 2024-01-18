#pragma once

#include "../LinearPrimitives.h"
#include "LossFunction.h"

namespace neural_net {

class MSE : LossFunction {
public:
    double Loss(const Vector& present, const Vector& expected) override;
    Vector LossGradient(const Vector& present, const Vector& expected) override;
};

}  // namespace neural_net
