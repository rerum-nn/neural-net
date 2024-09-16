#include "Metrics/Metric.h"
#include "Types.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

using neural_net::Matrix;
using neural_net::Metric;
using testing::FloatEq;

TEST(CategoricalAccuracyTest, General) {
    Matrix expected{{0, 0, 1}, {0, 1, 0}};
    Matrix pred{{0.1, 0.9, 0.8}, {0.05, 0.95, 0}};
    float accuracy = Metric::CategoricalAccuracy()(pred, expected);
    ASSERT_THAT(accuracy, FloatEq(0.5));
}

TEST(BinaryAccuracyTest, General) {
    Matrix expected{{1}, {1}, {0}, {0}};
    Matrix pred{{0.98}, {1}, {0}, {0.6}};
    float accuracy = Metric::BinaryAccuracy()(pred, expected);
    ASSERT_THAT(accuracy, FloatEq(0.75));
}

TEST(BinaryAccuracyTest, Threshold) {
    Metric bin1 = Metric::BinaryAccuracy(0.3);
    Metric bin2 = Metric::BinaryAccuracy(0.9);

    Matrix expected{{1}, {1}, {0}, {0}};
    Matrix pred{{0.98}, {0.5}, {0.1}, {0.2}};

    float accuracy1 = bin1(pred, expected);
    float accuracy2 = bin2(pred, expected);

    ASSERT_THAT(accuracy1, FloatEq(1.0));
    ASSERT_THAT(accuracy2, FloatEq(0.75));
}

TEST(BinaryPrecisionTest, General) {
    Matrix pred = Matrix{{1, 0, 1, 0}}.transpose();
    Matrix expected = Matrix{{0, 1, 1, 0}}.transpose();
    ASSERT_THAT(Metric::BinaryPrecision()(pred, expected), FloatEq(0.5));
}

TEST(BinaryPrecisionTest, Threshold) {
    Matrix pred = Matrix{{1, 0, 0.6, 0, 0.7}}.transpose();
    Matrix expected = Matrix{{0, 1, 1, 0, 1}}.transpose();
    ASSERT_THAT(Metric::BinaryPrecision(0.7)(pred, expected), FloatEq(0.5));
}

TEST(BinaryRecall, General) {
    Matrix pred = Matrix{{1, 0, 1, 0}}.transpose();
    Matrix expected = Matrix{{0, 1, 1, 0}}.transpose();
    ASSERT_THAT(Metric::BinaryRecall()(pred, expected), FloatEq(0.5));
}

TEST(BinaryRecall, Threshold) {
    Matrix pred = Matrix{{1, 0, 0.6, 0, 0.7}}.transpose();
    Matrix expected = Matrix{{0, 1, 1, 0, 1}}.transpose();
    ASSERT_THAT(Metric::BinaryRecall(0.7)(pred, expected), FloatEq(1./3));
}

TEST(F1ScoreTest, General) {
    Matrix pred = Matrix{{1, 0, 1, 0}}.transpose();
    Matrix expected = Matrix{{0, 1, 1, 0}}.transpose();
    ASSERT_THAT(Metric::BinaryF1Score()(pred, expected), FloatEq(0.5));
}

TEST(F1ScoreTest, Threshold) {
    Matrix pred = Matrix{{1, 0, 0.6, 0, 0.7}}.transpose();
    Matrix expected = Matrix{{0, 1, 1, 0, 1}}.transpose();
    ASSERT_THAT(Metric::BinaryF1Score(0.7)(pred, expected), FloatEq(0.4));
}
