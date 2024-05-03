#include "Tanh.h"

namespace neural_net {

Matrix Tanh::Apply(const Matrix& input_vector) {
    tanh_data_ = input_vector.unaryExpr([](float d) { return 2.f / (1.f + std::exp(-2*d)) - 1; });
    return tanh_data_;
}

Matrix Tanh::BackPropagation(const Matrix& loss) const {
    return 1 - tanh_data_.cwiseProduct(tanh_data_).array();
}

}  // namespace neural_net
