#pragma once

#include "Types.h"

#include <tuple>

namespace neural_net {

std::tuple<Matrix, Matrix, Matrix, Matrix> TrainTestSplit(const Matrix& data, const Matrix& labels,
                                                          double train_ratio = 0.8,
                                                          ShuffleMode mode = ShuffleMode::Shuffle);

Matrix IntLabelsToCategorical(const Matrix& labels);

}  // namespace neural_net
