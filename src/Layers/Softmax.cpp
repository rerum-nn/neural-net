#include "Softmax.h"

namespace neural_net {
Vector Softmax::Apply(const Vector& input_vector) {
    double mx = input_vector.maxCoeff();
    Vector transformed = (input_vector.array() - mx).array().exp();
    exp_vector_ = transformed / transformed.sum();
    return exp_vector_;
}

std::vector<ParametersGrad> Softmax::GetGradients(const RowVector& loss) {
    return {};
}

RowVector Softmax::BackPropagation(const RowVector& loss) const {
    Matrix delta(loss.rows(), loss.rows());
    for (Index i = 0; i < loss.rows(); ++i) {
        for (Index j = 0; j < loss.rows(); ++j) {
            double kron_delta = (i == j ? 1 : 0);
            delta(i, j) = exp_vector_[i] * (kron_delta - exp_vector_[j]);
        }
    }
    return loss * delta;
}
}  // namespace neural_net