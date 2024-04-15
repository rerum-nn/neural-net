#pragma once

#include "../Types.h"

namespace neural_net {

class Linear {
public:
    Linear(Index input, Index output);

    Vector Apply(const Vector& input_vector);
    RowVector Fit(const RowVector& loss);

private:
    Matrix weights_;
    Vector bias_;

    Vector input_vector_;
};

}  // namespace neural_net
