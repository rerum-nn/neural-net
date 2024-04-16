#pragma once

#include "Layers/Layer.h"
#include "LossFunctions/LossFunction.h"
#include "LossFunctions/MSE.h"
#include "Optimizers/Optimizer.h"
#include "Optimizers/SGD.h"
#include "Types.h"

#include <string>
#include <vector>

namespace neural_net {

class Network {
public:
    Network(std::initializer_list<Layer> layers);

    void Fit(const Matrix& input_data, const Matrix& labels, const LossFunction& loss = MSE(),
             Optimizer&& optimizer = SGD());
    Vector Predict(const Vector& input_vector);

    Network& AddLayer(const Layer& layer);
    Network& AddLayer(Layer&& layer);

    std::string Summary() const;

private:
    std::vector<Layer> layers_;
};

}  // namespace neural_net
