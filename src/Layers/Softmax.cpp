#include "Softmax.h"

namespace neural_net {
Matrix Softmax::Apply(const Matrix& input_vector) {
    assert(input_vector.size() > 0);
    double mx = input_vector.maxCoeff();
    Vector transformed = (input_vector.array() - mx).array().exp();
    exp_data_ = transformed / transformed.sum();
    return exp_data_;
}

std::vector<ParametersGrad> Softmax::GetGradients(const Matrix& loss) {
    return {};
}

Matrix Softmax::BackPropagation(const Matrix& loss) const {
    Matrix res(loss.rows(), loss.cols());
    for (Index i = 0; i < loss.rows(); ++i) {
        Matrix delta = exp_data_.row(i).asDiagonal().toDenseMatrix() -
                       exp_data_.row(i).transpose() * exp_data_.row(i);
        res.col(i) = loss * delta;
    }
    return res;
}
}  // namespace neural_net