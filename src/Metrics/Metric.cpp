#include "Metric.h"

namespace neural_net {

Metric::Metric(std::function<float(const Matrix &, const Matrix &)> func, const std::string &name)
    : compute_(func), name_(name) {
}

float neural_net::Metric::operator()(const Matrix &pred, const Matrix &expected) const {
    return compute_(pred, expected);
}

Metric Metric::BinaryAccuracy(float threshold) {
    assert(threshold <= 1 && threshold >= 0);
    auto func = [threshold](const Matrix &pred, const Matrix &expected) {
        assert(pred.rows() == expected.rows() && pred.cols() == 1 && expected.cols() == 1);
        Matrix rounded =
            pred.unaryExpr([threshold](float d) { return d >= threshold ? 1.f : 0.f; });
        Index right_answers = ((expected - rounded).array() == 0).count();
        return static_cast<float>(right_answers) / pred.rows();
    };

    return Metric(func, "bin_acc");
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
        return static_cast<float>(right_answers) / pred.rows();
    };

    return Metric(func, "cat_acc");
}

std::string Metric::GetName() const {
    return name_;
}

Metric Metric::BinaryPrecision(float threshold) {
    assert(threshold <= 1 && threshold >= 0);
    auto func = [threshold](const Matrix &pred, const Matrix &expected) {
        assert(pred.rows() == expected.rows() && pred.cols() == 1 && expected.cols() == 1);
        Matrix rounded =
            pred.unaryExpr([threshold](float d) { return d >= threshold ? 1.f : 0.f; });
        Index tp = ((expected + rounded).array() == 2).count();
        Index t = (rounded.array() == 1).count();
        if (t != 0) {
            return static_cast<float>(tp) / t;
        } else {
            return 0.f;
        }
    };
    return Metric(func, "bin_precision");
}

Metric Metric::BinaryRecall(float threshold) {
    assert(threshold <= 1 && threshold >= 0);
    auto func = [threshold](const Matrix &pred, const Matrix &expected) {
        assert(pred.rows() == expected.rows() && pred.cols() == 1 && expected.cols() == 1);
        Matrix rounded =
            pred.unaryExpr([threshold](float d) { return d >= threshold ? 1.f : 0.f; });
        Index tp = ((expected + rounded).array() == 2).count();
        Index t = (expected.array() == 1).count();
        if (t != 0) {
            return static_cast<float>(tp) / t;
        } else {
            return 0.f;
        }
    };
    return Metric(func, "bin_recall");
}

Metric Metric::BinaryF1Score(float threshold) {
    assert(threshold <= 1 && threshold >= 0);
    auto func = [threshold](const Matrix &pred, const Matrix &expected) {
        assert(pred.rows() == expected.rows() && pred.cols() == 1 && expected.cols() == 1);
        Matrix rounded =
            pred.unaryExpr([threshold](float d) { return d >= threshold ? 1.f : 0.f; });
        Index tp = ((expected + rounded).array() == 2).count();
        float precision = static_cast<float>(tp) / (rounded.array() == 1).count();
        float recall = static_cast<float>(tp) / (expected.array() == 1).count();
        float t = precision + recall;
        if (t == 0) {
            return 0.f;
        } else {
            return 2 * (precision * recall) / t;
        }
    };

    return Metric(func, "bin_f1-score");
}

}  // namespace neural_net
