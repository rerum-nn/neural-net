#pragma once

#include "Types.h"

namespace neural_net {

class Tanh {
public:
    Matrix Apply(const Matrix& input_vector);
    Matrix BackPropagation(const Matrix& loss) const;

    void Serialize(std::ostream& os) const;

private:
    Matrix tanh_data_;
};

}  // namespace neural_net
