#pragma once

#include "../Types.h"

namespace neural_net {

// TODO: rewrite to Type Erasure idiom
class LossFunction {
public:
    virtual double Loss(const Vector& present, const Vector& expected) = 0;
    virtual Vector LossGradient(const Vector& present, const Vector& expected) = 0;
};

}  // namespace neural_net
