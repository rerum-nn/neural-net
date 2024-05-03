#pragma once

#include "Types.h"

namespace neural_net {

class Softmax {
public:
    Matrix Apply(const Matrix& input_data);
    Matrix BackPropagation(const Matrix& loss) const;

    void Serialize(std::ostream& os) const;

private:
    Matrix exp_data_;
};

}  // namespace neural_net
