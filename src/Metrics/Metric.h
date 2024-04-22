#pragma once

#include "Types.h"

namespace neural_net {

class Metric {
public:
    static Metric BinaryAccuracy(double threshold = 0.5);
    static Metric CategoricalAccuracy();

    double operator()(const Matrix& pred, const Matrix& expected) const;

    std::string GetName() const;

private:
    Metric(std::function<double(const Matrix&, const Matrix&)> func, const std::string& name = "");

    std::function<double(const Matrix&, const Matrix&)> compute_;
    std::string name_;
};

}  // namespace neural_net
