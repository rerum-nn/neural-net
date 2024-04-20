#pragma once

#include "../Types.h"

#include <vector>

namespace neural_net {

class Linear {
public:
    Linear(Index input, Index output);
    Linear(Matrix weights, Vector bias);

    Vector Apply(const Vector& input_vector);
    std::vector<ParametersGrad> GetGradients(const RowVector& loss);
    RowVector BackPropagation(const RowVector& loss) const;

private:
    Matrix weights_;
    Matrix bias_;

    Matrix input_vector_;
};

}  // namespace neural_net
