#pragma once

#include "../Types.h"

namespace neural_net {

class Softmax {
public:
    Matrix Apply(const Matrix& input_vector);
    std::vector<ParametersGrad> GetGradients(const Matrix& loss);
    Matrix BackPropagation(const Matrix& loss) const;

private:
    Matrix exp_data_;
};

}  // namespace neural_net
