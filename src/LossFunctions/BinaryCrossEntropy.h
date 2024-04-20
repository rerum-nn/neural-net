#pragma once

#include "../Types.h"

namespace neural_net {

class BinaryCrossEntropy {
public:
    double Loss(const Vector& present, const Vector& expected) const;
    RowVector LossGradient(const Vector& present, const Vector& expected) const;

private:
    static constexpr double kEpsilon = 1e-8;
};

}  // namespace neural_net
