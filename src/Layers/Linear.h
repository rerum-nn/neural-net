#pragma once

#include "Types.h"

#include <vector>

namespace neural_net {

class Linear {
public:
    Linear() = default;
    Linear(Index input, Index output);
    Linear(Matrix weights, Vector bias);

    Matrix Apply(const Matrix& input_data);
    std::vector<ParametersGrad> GetGradients(const Matrix& loss);
    Matrix BackPropagation(const Matrix& loss) const;

    void Serialize(std::ostream& os) const;

    void SetWeights(const Matrix& weights, const Vector& bias);
    const Matrix& GetWeights() const;
    const Vector& GetBias() const;

private:
    Matrix weights_;
    Matrix bias_;

    Matrix input_data_;
};

}  // namespace neural_net
