#pragma once

#include "Layers/Layer.h"
#include "LossFunctions/LossFunction.h"
#include "LossFunctions/MSE.h"
#include "Metrics/Metric.h"
#include "Optimizers/Optimizer.h"
#include "Optimizers/SGD.h"
#include "Types.h"

#include <string>
#include <vector>

namespace neural_net {

struct FitParameters {
    LossFunction loss = MSE();
    Optimizer optimizer = SGD();
    size_t max_epoch = 10000;
    size_t batch_size = 1;
    double validate_ratio = 0.1;
    std::vector<Metric> metrics;
};

class Sequential {
public:
    Sequential(std::initializer_list<Layer> layers);

    Matrix Predict(const Matrix& input_data);

    std::vector<double> Fit(const Matrix& input_data, const Matrix& labels,
                            FitParameters fit_parameters);

    std::vector<double> Evaluate(const Matrix& input_data, const Matrix& labels,
                                 const LossFunction& loss, std::initializer_list<Metric> metrics);

    Sequential& AddLayer(const Layer& layer);
    Sequential& AddLayer(Layer&& layer);

    std::vector<Layer>& GetLayers();

private:
    std::vector<Layer> layers_;
};

}  // namespace neural_net
