#pragma once

#include "FitParameters.h"
#include "Layers/Layer.h"
#include "LossFunctions/LossFunction.h"
#include "Types.h"

#include <vector>

namespace neural_net {

class Network {
public:
    Network(std::initializer_list<Layer> layers);

    void Fit(const Matrix& input_data, const Matrix& expected_answers,
             const FitParameters& fit_parameters = {});
    Vector Predict(const Vector& input_vector);

private:
    std::vector<Layer> layers_;
};

}  // namespace neural_net
