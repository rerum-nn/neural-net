#include "Sigmoid.h"

#include <cmath>

namespace neural_net {

Matrix Sigmoid::Apply(const Matrix& input_vector) {
    input_data_ = input_vector;
    Matrix res = input_vector.unaryExpr([](double d) { return 1. / (1. + std::exp(-d)); });
    return res;
}

std::vector<ParametersGrad> Sigmoid::GetGradients(const Matrix& loss) {
    return {};
}

Matrix Sigmoid::BackPropagation(const Matrix& loss) const {
    Matrix derivative = input_data_.unaryExpr([](double d) {
        double sigmoid = 1. / (1. + std::exp(-d));
        return sigmoid * (1 - sigmoid);
    });
    return loss.cwiseProduct(derivative.transpose());
}

}  // namespace neural_net
