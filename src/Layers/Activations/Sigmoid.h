#pragma once

#include "Types.h"

namespace neural_net {

class Sigmoid {
public:
    Matrix Apply(const Matrix& input_data);
    Matrix BackPropagation(const Matrix& loss) const;

    void Serialize(std::ostream& os) const;

private:
    Matrix sigmoid_data_;
};

}  // namespace neural_net
