#pragma once

#include "Types.h"

#include <ostream>

namespace neural_net {

class ActivationNone {
public:
    Matrix Apply(const Matrix& input_data);
    Matrix BackPropagation(const Matrix& loss) const;

    void Serialize(std::ostream& os) const;
};

}  // namespace neural_net
