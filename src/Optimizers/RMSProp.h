#pragma once

#include "../LossFunctions/LossFunction.h"
#include "../Network.h"
#include "../Types.h"

namespace neural_net {

class RMSProp {
public:
    RMSProp(double lr = 0.03, double rho = 0);

    void operator()(Network& network, const Matrix& input_data, const Matrix& labels,
                    const LossFunction& loss, size_t max_epoch = 10000) const;

private:
    void UpdateParameter(const std::vector<ParametersGrad>& pack,
                         std::vector<Matrix>& old_grad) const;

    static constexpr double kEpsilon = 1e-7;

    double learning_rate_;
    double rho_;
};

}  // namespace neural_net
