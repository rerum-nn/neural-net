#pragma once

#include "../Types.h"

namespace neural_net {

class MSE {
public:
    Vector Loss(const Vector& present, const Vector& expected) const;
    RowVector LossGradient(const Vector& present, const Vector& expected) const;
};

}  // namespace neural_net
