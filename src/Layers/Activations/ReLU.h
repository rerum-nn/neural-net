#pragma once

#include "Types.h"

namespace neural_net {

class ReLU {
public:
    Matrix Apply(const Matrix& input_data);
    Matrix BackPropagation(const Matrix& loss) const;

    void Serialize(std::ostream& os) const;

private:
    Matrix computed_data_;
};

}  // namespace neural_net
