#include "ReLU.h"

namespace neural_net {

Matrix ReLU::Apply(const Matrix& input_data) {
    Matrix res = input_data.unaryExpr([](double d) { return std::max(0., d); });
    computed_data_ = res;
    return res;
}

std::vector<ParametersGrad> ReLU::GetGradients(const Matrix& loss) {
    return {};
}

Matrix ReLU::BackPropagation(const Matrix& loss) const {
    Matrix derivative = computed_data_.unaryExpr([](double d) { return d != 0 ? 1. : 0.; });
    return loss.cwiseProduct(derivative.transpose());
}
}  // namespace neural_net