#pragma once

#include "Types.h"

namespace neural_net {

class ReLU {
public:
    Matrix Apply(const Matrix& input_data);
    std::vector<ParametersGrad> GetGradients(const Matrix& loss);
    Matrix BackPropagation(const Matrix& loss) const;
private:
    Matrix computed_data_;
};

}  // namespace neural_net

