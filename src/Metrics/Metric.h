#pragma once

#include "Types.h"

namespace neural_net {

class Metric {
public:
    static Metric BinaryAccuracy(double threshold = 0.5);
    static Metric CategoricalAccuracy();

    double operator()(const Matrix& pred, const Matrix& expected) const;

private:
    Metric(std::function<double(const Matrix&, const Matrix&)> func);

    std::function<double(const Matrix&, const Matrix&)> compute_;
};

}  // namespace neural_net
