#include "ActivationNone.h"

namespace neural_net {
Matrix ActivationNone::Apply(const Matrix& input_data) {
    assert(input_data.size() > 0);
    return input_data;
}

Matrix ActivationNone::BackPropagation(const Matrix& loss) const {
    assert(loss.size() > 0);
    return loss;
}

void ActivationNone::Serialize(std::ostream& os) const {
    os << " none ";
}
}  // namespace neural_net
