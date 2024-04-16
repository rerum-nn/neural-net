#pragma once

#include "../Types.h"

#include <vector>

namespace neural_net {

class SGD {
public:
    SGD(double lr = 0.03);
    void Optimize(const std::vector<ParametersGrad>& params);

private:
    double learning_rate_;

};

}  // namespace neural_net
