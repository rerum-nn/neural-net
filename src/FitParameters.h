#pragma once

#include "LossFunctions/LossFunction.h"
#include "LossFunctions/MSE.h"
#include <cstddef>

namespace neural_net {

struct FitParameters {
    double learning_rate = 0.03;
    LossFunction loss_function = MSE();
    size_t max_epoch = 1000;
};


} // namespace neural_net

