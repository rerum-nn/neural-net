#pragma once

#include <Eigen/Dense>

namespace neural_net {

using Vector = Eigen::VectorXf;
using RowVector = Eigen::RowVectorXf;
using Matrix = Eigen::MatrixXf;
using Index = Eigen::Index;
using Array = Eigen::ArrayXXf;
using PermutationMatrix = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

struct UpdatePack {
    Matrix& weights;
    Matrix weights_grad;
    Vector& bias;
    Vector bias_grad;
};

struct GradsPack {
    Matrix weights_grad;
    Vector bias_grad;
};

enum class ShuffleMode { Static, Shuffle };

}  // namespace neural_net
