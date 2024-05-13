#include "LossFunctions/BinaryCrossEntropy.h"
#include "LossFunctions/CategoricalCrossEntropy.h"
#include "LossFunctions/LossFunction.h"
#include "LossFunctions/MAE.h"
#include "LossFunctions/MSE.h"
#include "TestHelpers.h"
#include "Types.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

using neural_net::LossFunction;
using neural_net::Matrix;
using testing::FloatEq;

class BinaryCrossEntropyTest : public testing::Test {
protected:
    LossFunction bce_ = neural_net::BinaryCrossEntropy();
    Matrix present_ = Matrix{{0.2, 0.7, 0.9}};
    Matrix expected_ = Matrix{{0, 1, 1}};
};

TEST_F(BinaryCrossEntropyTest, Loss) {
    ASSERT_THAT(bce_->Loss(present_, expected_), FloatEq(0.2231435328722));
}

TEST_F(BinaryCrossEntropyTest, LossGradient) {
    CheckCloseMatrix(bce_->LossGradient(present_, expected_),
                     Matrix{{1.25, -1.42857146, -1.111111111}});
}

class CategoricalCrossEntropyTest : public testing::Test {
protected:
    LossFunction cce_ = neural_net::CategoricalCrossEntropy();
    Matrix present_ = Matrix{{0.4, 0.3, 0.7}, {0.8, 0.1, 0}, {0.3, 0.3, 1}, {0.4, 0.1, 0.6}};
    Matrix expected_ = Matrix{
        {1, 0, 0},
        {1, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
    };
};

TEST_F(CategoricalCrossEntropyTest, Loss) {
    ASSERT_THAT(cce_->Loss(present_, expected_), FloatEq(0.8605045));
}

TEST_F(CategoricalCrossEntropyTest, LossGradient) {
    CheckCloseMatrix(cce_->LossGradient(present_, expected_),
                     Matrix{{-0.625, 0, 0}, {-0.3125, 0, 0}, {0, 0, -0.25}, {0, -2.5, 0}}, 1e-5);
}

class MSETest : public testing::Test {
protected:
    LossFunction mse_ = neural_net::MSE();
    Matrix present_ = Matrix{{4, 8, 12}, {8, 1, 3}};
    Matrix expected_ = Matrix{{1, 9, 2}, {-5, -2, 6}};
};

TEST_F(MSETest, Loss) {
    ASSERT_THAT(mse_->Loss(present_, expected_), FloatEq(49.5));
}

TEST_F(MSETest, LossGradient) {
    CheckCloseMatrix(mse_->LossGradient(present_, expected_),
                     Matrix{{-1, 0.333, -3.333}, {-4.333, -1, 1}}, 1e-3);
}

class MAETest : public testing::Test {
protected:
    LossFunction mae_ = neural_net::MAE();
    Matrix present_ = Matrix{{4, 8, 12}, {8, 1, 3}};
    Matrix expected_ = Matrix{{1, 9, 2}, {-5, -2, 6}};
};

TEST_F(MAETest, Loss) {
    ASSERT_THAT(mae_->Loss(present_, expected_), FloatEq(5.5));
}

TEST_F(MAETest, LossGradient) {
    CheckCloseMatrix(mae_->LossGradient(present_, expected_),
                     Matrix{{1. / 6, -1. / 6, 1. / 6}, {1. / 6, 1. / 6, -1. / 6}});
}
