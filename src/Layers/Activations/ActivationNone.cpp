#include "ActivationNone.h"

namespace neural_net {
Matrix ActivationNone::Apply(const Matrix& input_data) {
    return input_data;
}

Matrix ActivationNone::BackPropagation(const Matrix& loss) const {
    return loss;
}

void ActivationNone::Serialize(std::ostream& os) const {
    os << " none ";
}
}  // namespace neural_net
