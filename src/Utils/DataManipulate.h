#pragma once

#include "Layers/Linear.h"
#include "Types.h"

#include <tuple>

namespace neural_net {

std::tuple<Matrix, Matrix, Matrix, Matrix> TrainTestSplit(const Matrix& data, const Matrix& labels,
                                                          double train_ratio = 0.8,
                                                          ShuffleMode mode = ShuffleMode::Shuffle);

Matrix IntLabelsToCategorical(const Matrix& labels);
Linear DeserializeLayer(std::istream& is);

}  // namespace neural_net
