#include "Metric.h"

namespace neural_net {

Metric::Metric(std::function<double(const Matrix &, const Matrix &)> func, const std::string &name)
    : compute_(func), name_(name) {
}

double neural_net::Metric::operator()(const Matrix &pred, const Matrix &expected) const {
    return compute_(pred, expected);
}

Metric Metric::BinaryAccuracy(double threshold) {
    assert(threshold <= 1 && threshold >= 0);
    auto func = [threshold](const Matrix &pred, const Matrix &expected) {
        assert(pred.rows() == expected.rows() && pred.cols() == 1 && expected.cols() == 1);
        Matrix rounded = pred.unaryExpr([threshold](float d) { return d > threshold ? 1.f : 0.f; });
        Index right_answers = ((expected - rounded).array() == 0).count();
        return static_cast<double>(right_answers) / pred.rows();
    };

    return Metric(func, "binary_acc");
}

Metric Metric::CategoricalAccuracy() {
    auto argmax = [](const Vector &vector) {
        Index i = 0;
        vector.maxCoeff(&i);
        Vector res(vector.size());
        res.setZero();
        res[i] = 1;
        return res;
    };

    auto func = [argmax](const Matrix &pred, const Matrix &expected) {
        assert(pred.rows() == expected.rows() && pred.cols() == expected.cols());
        Matrix rounded(pred.rows(), pred.cols());
        for (Index i = 0; i < pred.rows(); ++i) {
            rounded.row(i) = argmax(pred.row(i));
        }
        Index right_answers = ((expected - rounded).rowwise().norm().array() == 0.).count();
        return static_cast<double>(right_answers) / pred.rows();
    };

    return Metric(func, "categorical_acc");
}

std::string Metric::GetName() const {
    return name_;
}

}  // namespace neural_net
