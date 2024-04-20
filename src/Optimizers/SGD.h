#pragma once

#include "../LossFunctions/LossFunction.h"
#include "../Sequential.h"
#include "../Types.h"

#include <vector>

namespace neural_net {

class SGD {
public:
    SGD(double lr = 0.01, double momentum = .0);

    void operator()(Sequential& network, const Matrix& input_data, const Matrix& labels,
                    const LossFunction& loss, size_t max_epoch = 10000) const;

private:
    void UpdateParameter(const std::vector<ParametersGrad>& pack,
                         std::vector<Matrix>& old_grad) const;

    double learning_rate_;
    double moment_;
};

}  // namespace neural_net
