#pragma once

#include "../Types.h"

namespace neural_net {

class Sigmoid {
public:
    Vector Apply(const Vector& input_vector);
    std::vector<ParametersGrad> GetGradients(const RowVector& loss);
    RowVector BackPropagation(const RowVector& loss) const;

private:
    Vector input_vector_;
};

}  // namespace neural_net
