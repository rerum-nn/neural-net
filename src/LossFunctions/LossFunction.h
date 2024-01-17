#pragma once

#include "../LinearPrimitives.h"

namespace neural_net {

// TODO: rewrite to Type Erasure idiom
class LossFunction {
public:
    virtual double Loss(const DataVector& present, const DataVector& expected) = 0;
    virtual DataVector LossGradient(const DataVector& present, const DataVector& expected) = 0;
};

}
