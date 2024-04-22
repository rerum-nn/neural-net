#include "Metrics/Metric.h"
#include "Types.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

using neural_net::Matrix;
using neural_net::Metric;
using testing::DoubleEq;

TEST(CategoricalAccuracy, General) {
    Matrix expected{{0, 0, 1}, {0, 1, 0}};
    Matrix pred{{0.1, 0.9, 0.8}, {0.05, 0.95, 0}};
    double accuracy = Metric::CategoricalAccuracy()(pred, expected);
    ASSERT_THAT(accuracy, DoubleEq(0.5));
}

TEST(BinaryAccuracy, General) {
    Matrix expected{{1}, {1}, {0}, {0}};
    Matrix pred{{0.98}, {1}, {0}, {0.6}};
    double accuracy = Metric::BinaryAccuracy()(pred, expected);
    ASSERT_THAT(accuracy, DoubleEq(0.75));
}

TEST(BinaryAccuracy, Threshold) {
    Metric bin1 = Metric::BinaryAccuracy(0.3);
    Metric bin2 = Metric::BinaryAccuracy(0.9);

    Matrix expected{{1}, {1}, {0}, {0}};
    Matrix pred{{0.98}, {0.5}, {0.1}, {0.2}};

    double accuracy1 = bin1(pred, expected);
    double accuracy2 = bin2(pred, expected);

    ASSERT_THAT(accuracy1, DoubleEq(1.0));
    ASSERT_THAT(accuracy2, DoubleEq(0.75));
}

TEST(BinaryAccuracy, InvalidThreshold) {
    ASSERT_DEATH(Metric::BinaryAccuracy(-0.5), "");
    ASSERT_DEATH(Metric::BinaryAccuracy(1.5), "");
}
