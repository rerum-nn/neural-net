#include "Metric.h"

neural_net::Metric::Metric(std::function<double(const Matrix &, const Matrix &)> func)
    : compute_(func) {
}

double neural_net::Metric::operator()(const neural_net::Matrix &pred,
                                       const neural_net::Matrix &expected) const {
    return compute_(pred, expected);
}

neural_net::Metric neural_net::Metric::BinaryAccuracy(double threshold) {
    auto func = [threshold](const Matrix &pred, const Matrix &expected) {
        assert(pred.rows() == expected.rows() && pred.cols() == 1 && expected.cols() == 1);
        Matrix rounded = pred.unaryExpr([threshold](double d) { return d > threshold ? 1. : 0.; });
        Index right_answers = ((expected - pred).array() == 0).count();
        return right_answers / pred.rows();
    };

    return Metric(func);
}

neural_net::Metric neural_net::Metric::CategoricalAccuracy() {
    auto argmax = [](const Vector& vector) {
        double max_coeff = vector.maxCoeff();
        Index i = 0;
        for (; i < vector.size(); ++i) {
            if (vector[i] == max_coeff) {
                break;
            }
        }
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
        Index right_answers = ((expected - pred).rowwise().norm().array() == 0).count();
        return static_cast<double>(right_answers) / pred.rows();
    };

    return Metric(func);
}
