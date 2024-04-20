#pragma once

#include "../LossFunctions/LossFunction.h"
#include "../Sequential.h"
#include "../Types.h"

#include <memory>
#include <vector>

namespace neural_net {

using Optimizer =
    std::function<void(Sequential&, const Matrix&, const Matrix&, const LossFunction&, size_t)>;

class Optimizers {
public:
    static Optimizer SGD(double lr = 0.01);
    static Optimizer Momentum(double lr = 0.01, double momentum = .0);

    static Optimizer AdaGrad(double lr = 0.01);
    static Optimizer RMSProp(double lr = 0.001, double rho = 0.9);
    static Optimizer Adam(double lr = 0.001, double beta_1 = 0.9, double beta_2 = 0.999);
};

}  // namespace neural_net
