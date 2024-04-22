#include "Linear.h"

#include "Utils/Random.h"

namespace neural_net {

Linear::Linear(Index input, Index output)
    : weights_(Random::Normal(output, input)), bias_(Random::Normal(output, 1)) {
    assert(input > 0 && "input size of layer should be positive");
    assert(output > 0 && "output size of layer should be positive");
}

Linear::Linear(Matrix weights, Vector bias) : weights_(weights), bias_(bias) {
    assert(weights.size() > 0 && "weights cannot be empty");
    assert(weights.rows() == bias.rows() && "weights rows should be equal bias rows");
}

Matrix Linear::Apply(const Matrix& input_data) {
    assert(input_data.rows() == weights_.cols() &&
           "input vector size and layer input size are not consisted");
    input_data_ = input_data;
    return weights_ * input_data + bias_.replicate(1, input_data.cols());
}

std::vector<ParametersGrad> Linear::GetGradients(const Matrix& loss) {
    assert(input_data_.size() != 0 && "input_vector for fit information hasn't been transferred");
    return {{weights_, (input_data_ * loss).transpose()},
            {bias_, loss.transpose().rowwise().sum()}};
}

Matrix Linear::BackPropagation(const Matrix& loss) const {
    return loss * weights_;
}

}  // namespace neural_net
