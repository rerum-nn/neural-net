#pragma once

#include "Types.h"

namespace neural_net {

class MAE {
public:
    double Loss(const Matrix& present, const Matrix& expected) const;
    Matrix LossGradient(const Matrix& present, const Matrix& expected) const;
};

}  // namespace neural_net
