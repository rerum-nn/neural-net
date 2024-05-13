#include "Sigmoid.h"

#include <cmath>

namespace neural_net {

Matrix Sigmoid::Apply(const Matrix& input_data) {
    assert(input_data.size() > 0);
    sigmoid_data_ = input_data.unaryExpr([](float d) { return 1.f / (1.f + std::exp(-d)); });
    return sigmoid_data_;
}

Matrix Sigmoid::BackPropagation(const Matrix& loss) const {
    assert(loss.size() > 0);
    Matrix derivative = sigmoid_data_.array() * (1 - sigmoid_data_.array());
    return loss.cwiseProduct(derivative.transpose());
}

void Sigmoid::Serialize(std::ostream& os) const {
    os << " sigmoid ";
}

}  // namespace neural_net
