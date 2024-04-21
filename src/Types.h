#pragma once

#include <Eigen/Dense>

#include <vector>

namespace neural_net {

using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Matrix = Eigen::MatrixXd;
using Index = Eigen::Index;
using Array = Eigen::ArrayXXd;
using PermutationMatrix = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

struct ParametersGrad {
    Matrix& param;
    Matrix grad;
};

enum class ShuffleMode {
    Static,
    Shuffle
};

}  // namespace neural_net
