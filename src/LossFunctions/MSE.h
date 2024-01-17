#pragma once

#include "../LinearPrimitives.h"
#include "LossFunction.h"

namespace neural_net {

class MSE : LossFunction {
public:
    double Loss(const DataVector& present, const DataVector& expected) override;
    DataVector LossGradient(const DataVector& present, const DataVector& expected) override;
};

}  // namespace neural_net
