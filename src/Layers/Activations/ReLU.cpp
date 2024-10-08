#include "ReLU.h"

namespace neural_net {

Matrix ReLU::Apply(const Matrix& input_data) {
    assert(input_data.size() > 0);
    return computed_data_ = input_data.unaryExpr([](float d) { return std::max(0.f, d); });
}

Matrix ReLU::BackPropagation(const Matrix& loss) const {
    assert(loss.size() > 0);
    Matrix derivative = computed_data_.unaryExpr([](float d) { return d > 0 ? 1.f : 0.f; });
    return loss.cwiseProduct(derivative.transpose());
}

void ReLU::Serialize(std::ostream& os) const {
    os << " relu ";
}

}  // namespace neural_net
