#include "Sigmoid.h"

#include <cmath>

namespace neural_net {

Matrix Sigmoid::Apply(const Matrix& input_vector) {
    Matrix res = input_vector.unaryExpr([](double d) { return 1. / (1. + std::exp(-d)); });
    sigmoid_data_ = res;
    return res;
}

std::vector<ParametersGrad> Sigmoid::GetGradients(const Matrix& loss) {
    return {};
}

Matrix Sigmoid::BackPropagation(const Matrix& loss) const {
    Matrix derivative = sigmoid_data_.array() * (1 - sigmoid_data_.array());
    return loss.cwiseProduct(derivative.transpose());
}

}  // namespace neural_net
