#include "Linear.h"

#include "Utils/Random.h"

#include <iostream>

namespace neural_net {

Linear::Linear(Index input, Index output, Activation&& activation)
    : weights_(Random::Normal(output, input)),
      bias_(Random::Normal(output, 1)),
      activation_(std::move(activation)) {
    assert(input > 0 && "input size of layer should be positive");
    assert(output > 0 && "output size of layer should be positive");
}

Linear::Linear(Matrix weights, Vector bias, Activation&& activation)
    : weights_(weights), bias_(bias), activation_(std::move(activation)) {
    assert(weights.size() > 0 && "weights cannot be empty");
    assert(weights.rows() == bias.rows() && "weights rows should be equal bias rows");
}

Matrix Linear::Apply(const Matrix& input_data) {
    assert(input_data.rows() == weights_.cols() &&
           "input vector size and layer input size are not consisted");
    input_data_ = input_data;
    return activation_->Apply(weights_ * input_data + bias_.replicate(1, input_data.cols()));
}

UpdatePack Linear::GetGradients(const Matrix& loss) {
    assert(input_data_.size() != 0 && "input_vector for fit information hasn't been transferred");
    assert(input_data_.cols() == loss.rows());
    return {weights_, (input_data_ * loss).transpose(), bias_, loss.transpose().rowwise().sum()};
}

Matrix Linear::BackPropagation(const Matrix& loss) const {
    assert(loss.cols() == weights_.rows());
    return loss * weights_;
}

Matrix Linear::BackPropagationActivation(const Matrix& loss) const {
    return activation_->BackPropagation(loss);
}

void Linear::SetWeights(const Matrix& weights, const Vector& bias) {
    assert(weights.size() > 0);
    assert(weights.rows() == bias.rows());
    weights_ = weights;
    bias_ = bias;
    input_data_ = Matrix(0, 0);
}

const Matrix& Linear::GetWeights() const {
    return weights_;
}

const Vector& Linear::GetBias() const {
    return bias_;
}

void Linear::Serialize(std::ostream& os) const {
    os << " linear " << weights_.rows() << ' ' << weights_.cols() << weights_ << bias_;
    activation_->Serialize(os);
}

}  // namespace neural_net
