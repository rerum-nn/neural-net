#include "BinaryCrossEntropy.h"

namespace neural_net {

float BinaryCrossEntropy::Loss(const Matrix& present, const Matrix& expected) const {
    assert(present.rows() == expected.rows());
    assert(present.cols() == 1 && expected.cols() == 1 &&
           "it should be output layer size equal 1 for binarycrossentropy loss function");
    Array present_array = present.array();
    Array expected_array = expected.array();
    Vector entropy = expected_array * (present_array + kEpsilon).log() +
                     (1 - expected_array) * (1 - present_array + kEpsilon).log();
    return -entropy.sum() / present.rows();
}

Matrix BinaryCrossEntropy::LossGradient(const Matrix& present, const Matrix& expected) const {
    assert(present.rows() == expected.rows());
    assert(present.cols() == 1 && expected.cols() == 1 &&
           "it should be output layer size equal 1 for binarycrossentropy loss function");
    Array present_array = present.array();
    Array expected_array = expected.array();
    return -(expected_array / present_array - (1 - expected_array) / (1 - present_array)) /
           present.rows();
}

}  // namespace neural_net
