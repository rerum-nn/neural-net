#include "Tanh.h"

namespace neural_net {

Matrix Tanh::Apply(const Matrix& input_data) {
    assert(input_data.size() > 0);
    tanh_data_ = input_data.unaryExpr([](float d) { return 2.f / (1.f + std::exp(-2*d)) - 1; });
    return tanh_data_;
}

Matrix Tanh::BackPropagation(const Matrix& loss) const {
    assert(loss.size() > 0);
    return 1 - tanh_data_.cwiseProduct(tanh_data_).array();
}

void Tanh::Serialize(std::ostream& os) const {
    os << " tanh ";
}

}  // namespace neural_net
