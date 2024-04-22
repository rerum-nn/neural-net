#include "Softmax.h"

namespace neural_net {

Matrix Softmax::Apply(const Matrix& input_vector) {
    assert(input_vector.size() > 0);
    double mx = input_vector.maxCoeff();
    Matrix transformed = (input_vector.array() - mx).exp();
    Vector sums = transformed.colwise().sum();
    exp_data_.resize(input_vector.rows(), input_vector.cols());
    for (Index i = 0; i < input_vector.cols(); ++i) {
        exp_data_.col(i) = transformed.col(i) / sums[i];
    }
    return exp_data_;
}

std::vector<ParametersGrad> Softmax::GetGradients(const Matrix& loss) {
    return {};
}

Matrix Softmax::BackPropagation(const Matrix& loss) const {
    Matrix res(loss.rows(), loss.cols());
    for (Index i = 0; i < loss.rows(); ++i) {
        Matrix delta = exp_data_.col(i).asDiagonal().toDenseMatrix() -
                       exp_data_.col(i) * exp_data_.col(i).transpose();
        res.row(i) = loss.row(i) * delta;
    }
    return res;
}

}  // namespace neural_net
