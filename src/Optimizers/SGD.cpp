#include "SGD.h"

namespace neural_net {
SGD::SGD(double lr) : learning_rate_(lr) {
}

void SGD::Optimize(const std::vector<ParametersGrad>& params) {
    for (const ParametersGrad& param : params) {
        param.param -= learning_rate_ * param.grad;
    }
}
}  // namespace neural_net