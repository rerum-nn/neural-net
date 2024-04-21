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

class Sequential {
public:
    Sequential(std::initializer_list<Layer> layers);

    Matrix Predict(const Matrix& input_data);

    std::vector<double> Fit(const Matrix& input_data, const Matrix& labels,
                            const LossFunction& loss = MSE(), Optimizer&& optimizer = SGD(), size_t max_epoch = 10000);

    Sequential& AddLayer(const Layer& layer);
    Sequential& AddLayer(Layer&& layer);

    std::vector<Layer>& GetLayers();

private:
    std::vector<Layer> layers_;
};

}  // namespace neural_net
