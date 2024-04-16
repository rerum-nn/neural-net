#include "BinaryCrossEntropy.h"

namespace neural_net {
Vector BinaryCrossEntropy::Loss(const Vector& present, const Vector& expected) const {
    assert(present.size() == 1 && expected.size() == 1 &&
           "it should be output layer size equal 1 for binarycrossentropy loss function");
    return Vector{{-expected[0] * std::log(present[0]) + (1 - expected[0]) * std::log(1 - present[0])}};
}

RowVector BinaryCrossEntropy::LossGradient(const Vector& present, const Vector& expected) const {
    assert(present.size() == 1 && expected.size() == 1 &&
           "it should be output layer size equal 1 for binarycrossentropy loss function");
    return Vector{{-(expected[0]/present[0] - (1 - expected[0])/(1-present[0]))}};
}

}  // namespace neural_net