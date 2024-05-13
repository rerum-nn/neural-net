#include "Layers/Activations/Activation.h"
#include "Layers/Activations/ReLU.h"
#include "Layers/Activations/Sigmoid.h"
#include "Layers/Activations/Softmax.h"
#include "Layers/Linear.h"
#include "TestHelpers.h"
#include "Types.h"

#include <gtest/gtest.h>

using neural_net::Activation;
using neural_net::ActivationNone;
using neural_net::Index;
using neural_net::Linear;
using neural_net::Matrix;
using neural_net::ReLU;
using neural_net::Sigmoid;
using neural_net::Softmax;
using neural_net::Vector;
using neural_net::UpdatePack;

class LinearTest : public testing::Test {
protected:
    Linear linear_layer_ = Linear(Matrix{{1.0, 3.0}, {-2.0, -4.0}}, Vector{{5.0, -6.0}});
};

TEST_F(LinearTest, Correctness) {
    Matrix input{{-1.0, 2.0}};
    input = linear_layer_.Apply(input.transpose());
    Activation relu = ReLU();
    input = relu->Apply(input);
    CheckCloseMatrix(input.transpose(), Matrix{{10.0, 0.0}});

    linear_layer_ = Linear(Matrix{{1.0, 3.0}, {-2.0, -4.0}}, Vector{{0., 0.}});
    input = Matrix{{-1.0, 2.0}}.transpose();
    input = linear_layer_.Apply(input);
    CheckCloseMatrix(input.transpose(), Matrix{{5.0, -6.0}});
}

TEST_F(LinearTest, GetGradients) {
    Matrix input{{-1.0, 2.0}, {3.0, -4.0}};
    linear_layer_.Apply(input);
    Matrix loss(2, 2);
    loss << 0.1, 0.2, 0.3, 0.4;
    UpdatePack gradients = linear_layer_.GetGradients(loss);

    EXPECT_EQ(gradients.weights, linear_layer_.GetWeights());
    EXPECT_EQ(gradients.bias, linear_layer_.GetBias());
    CheckCloseMatrix(gradients.weights_grad, Matrix{{0.5, -0.9}, {0.6, -1.0}});
    CheckCloseMatrix(gradients.bias_grad, Vector{{0.4, 0.6}});
}

TEST_F(LinearTest, BackPropagation) {
    Matrix loss{{0.1, 0.2}, {0.3, 0.4}};
    CheckCloseMatrix(linear_layer_.BackPropagation(loss), Matrix{{-0.3, -0.5}, {-0.5, -0.7}});
}

TEST(ActivationNoneTest, General) {
    Activation activation = ActivationNone();

    Matrix matrix = Matrix::Random(3, 3);
    CheckCloseMatrix(activation->Apply(matrix), matrix);
    CheckCloseMatrix(activation->BackPropagation(matrix), matrix);
}

class ReLUTest : public testing::Test {
protected:
    Activation relu_ = ReLU();
    Matrix input_ = Matrix{{-10, -5, 0, 5, 10}};
};

TEST_F(ReLUTest, Apply) {
    CheckCloseMatrix(relu_->Apply(input_.transpose()).transpose(), Matrix{{0, 0, 0, 5, 10}});
}

TEST_F(ReLUTest, BackPropagation) {
    relu_->Apply(input_.transpose());
    Matrix loss{{0, -1, 2, -6, 102}};
    CheckCloseMatrix(relu_->BackPropagation(loss), Matrix{{0, 0, 0, -6, 102}});
}

TEST(Softmax, Correctness) {
    Activation softmax = Softmax();
    Matrix input{{1, 2, 1}, {1, 2, 1}};
    input = softmax->Apply(input.transpose());
    CheckCloseMatrix(input.transpose(), Matrix{{0.21194157, 0.5761169, 0.21194157},
                                               {0.21194157, 0.5761169, 0.21194157}});
}
