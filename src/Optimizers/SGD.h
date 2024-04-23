#pragma once

#include "Layers/Linear.h"
#include "LossFunctions/LossFunction.h"
#include "Types.h"

#include <vector>

namespace neural_net {

class SGD {
public:
    SGD(double lr = 0.01, double momentum = .0);

    void InitParameters(const std::vector<Linear>& layers);
    void Update(const std::vector<ParametersGrad>& pack, size_t layer_id);
    void BatchCallback();
    void EpochCallback(size_t epoch, size_t max_epoch);

private:
    double learning_rate_;
    double moment_;

    std::vector<std::vector<Matrix>> old_grads_;
};

}  // namespace neural_net
