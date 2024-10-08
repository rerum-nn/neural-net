#pragma once

#include "Activations/Activation.h"
#include "Activations/ActivationNone.h"
#include "Types.h"

#include <vector>

namespace neural_net {

class Linear {
public:
    Linear(Index input, Index output, Activation&& activation = ActivationNone());
    Linear(Matrix weights, Vector bias, Activation&& activation = ActivationNone());

    Matrix Apply(const Matrix& input_data);
    UpdatePack GetGradients(const Matrix& loss);
    Matrix BackPropagation(const Matrix& loss) const;
    Matrix BackPropagationActivation(const Matrix& loss) const;

    void Serialize(std::ostream& os) const;

    void SetWeights(const Matrix& weights, const Vector& bias);
    const Matrix& GetWeights() const;
    const Vector& GetBias() const;

private:
    Matrix weights_;
    Vector bias_;

    Matrix input_data_;

    Activation activation_;
};

}  // namespace neural_net
