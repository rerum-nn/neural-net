#include "Sigmoid.h"

#include <cmath>

namespace neural_net {

Matrix Sigmoid::Apply(const Matrix& input_vector) {
    Matrix res = input_vector.unaryExpr([](float d) { return 1.f / (1.f + std::exp(-d)); });
    sigmoid_data_ = res;
    return res;
}

Matrix Sigmoid::BackPropagation(const Matrix& loss) const {
    Matrix derivative = sigmoid_data_.array() * (1 - sigmoid_data_.array());
    return loss.cwiseProduct(derivative.transpose());
}

void Sigmoid::Serialize(std::ostream& os) const {
    os << " sigmoid ";
}

}  // namespace neural_net
