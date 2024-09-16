#pragma once

#include "Types.h"

namespace neural_net {

class Metric {
public:
    static Metric BinaryAccuracy(float threshold = 0.5);
    static Metric CategoricalAccuracy();

    static Metric BinaryPrecision(float threshold = 0.5);
    static Metric BinaryRecall(float threshold = 0.5);
    static Metric BinaryF1Score(float threshold = 0.5);

    float operator()(const Matrix& pred, const Matrix& expected) const;

    std::string GetName() const;

private:
    Metric(std::function<float(const Matrix&, const Matrix&)> func, const std::string& name = "");

    std::function<float(const Matrix&, const Matrix&)> compute_;
    std::string name_;
};

}  // namespace neural_net
