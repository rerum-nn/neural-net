#pragma once

#include "../Types.h"

namespace neural_net {

class Sigmoid {
public:
    Vector Apply(const Vector& input_vector);
    RowVector Fit(const RowVector& loss);

private:
    Vector input_vector_;
};

}  // namespace neural_net
