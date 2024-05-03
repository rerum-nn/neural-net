#include "Softmax.h"

namespace neural_net {

Matrix Softmax::Apply(const Matrix& input_data) {
    assert(input_data.size() > 0);
    double mx = input_data.maxCoeff();
    Matrix transformed = (input_data.array() - mx).exp();
    Vector sums = transformed.colwise().sum();
    exp_data_.resize(input_data.rows(), input_data.cols());
    for (Index i = 0; i < input_data.cols(); ++i) {
        exp_data_.col(i) = transformed.col(i) / sums[i];
    }
    return exp_data_;
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

void Softmax::Serialize(std::ostream& os) const {
    os << " softmax ";
}

}  // namespace neural_net
