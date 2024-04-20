#pragma once

#include <Eigen/Dense>

#include <vector>

namespace neural_net {

using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Matrix = Eigen::MatrixXd;
using Index = Eigen::Index;

struct ParametersGrad {
    Matrix& param;
    Matrix grad;
};

}  // namespace neural_net
