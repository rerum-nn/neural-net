#pragma once

#include "Types.h"

namespace neural_net {

class CategoricalCrossEntropy {
public:
    double Loss(const Matrix& present, const Matrix& expected) const;
    Matrix LossGradient(const Matrix& present, const Matrix& expected) const;

private:
    static constexpr double kEpsilon = 1e-7;
};

}  // namespace neural_net
