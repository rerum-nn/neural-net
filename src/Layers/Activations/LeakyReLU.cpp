#include "LeakyReLU.h"

namespace neural_net {

LeakyReLU::LeakyReLU(float alpha) : alpha_(alpha) {
}

Matrix LeakyReLU::Apply(const Matrix& input_data) {
    assert(input_data.size() > 0);
    return computed_data_ =
               input_data.unaryExpr([this](float d) { return std::max(alpha_ * d, d); });
}

Matrix LeakyReLU::BackPropagation(const Matrix& loss) const {
    assert(loss.size() > 0);
    Matrix derivative = computed_data_.unaryExpr([this](float d) { return d >= 0 ? 1.f : alpha_; });
    return loss.cwiseProduct(derivative.transpose());
}

void LeakyReLU::Serialize(std::ostream& os) const {
    os << " leakyrelu ";
}

}  // namespace neural_net
