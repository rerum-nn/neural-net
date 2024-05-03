#pragma once

#include "Types.h"

namespace neural_net {

class LeakyReLU {
public:
    LeakyReLU(float alpha = 0.01);

    Matrix Apply(const Matrix& input_data);
    Matrix BackPropagation(const Matrix& loss) const;

    void Serialize(std::ostream& os) const;

private:
    float alpha_;
    Matrix computed_data_;
};

}  // namespace neural_net
