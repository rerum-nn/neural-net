#include "Layers/Activations/Activation.h"
#include "Layers/Activations/LeakyReLU.h"
#include "Layers/Activations/ReLU.h"
#include "Layers/Activations/Sigmoid.h"
#include "Layers/Activations/Softmax.h"
#include "Layers/Activations/Tanh.h"
#include "Layers/Linear.h"
#include "TestHelpers.h"
#include "Types.h"

#include <gtest/gtest.h>

using neural_net::Activation;
using neural_net::Matrix;
using neural_net::UpdatePack;
using neural_net::Vector;

class LinearTest : public testing::Test {
protected:
    neural_net::Linear linear_layer_ =
        neural_net::Linear(Matrix{{1.0, 3.0}, {-2.0, -4.0}}, Vector{{5.0, -6.0}});
};

TEST_F(LinearTest, Correctness) {
    Matrix input{{-1.0, 2.0}};
    input = linear_layer_.Apply(input.transpose());
    Activation relu = neural_net::ReLU();
    input = relu->Apply(input);
    CheckCloseMatrix(input.transpose(), Matrix{{10.0, 0.0}});

    linear_layer_ = neural_net::Linear(Matrix{{1.0, 3.0}, {-2.0, -4.0}}, Vector{{0., 0.}});
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
    Activation activation = neural_net::ActivationNone();

    Matrix matrix = Matrix::Random(3, 3);
    CheckCloseMatrix(activation->Apply(matrix), matrix);
    CheckCloseMatrix(activation->BackPropagation(matrix), matrix);
}

class ReLUTest : public testing::Test {
protected:
    Activation relu_ = neural_net::ReLU();
    Matrix input_ = Matrix{{-10, -5, 0, 5, 10}};
};

TEST_F(ReLUTest, Apply) {
    CheckCloseMatrix(relu_->Apply(input_), Matrix{{0, 0, 0, 5, 10}});
}

TEST_F(ReLUTest, BackPropagation) {
    relu_->Apply(input_.transpose());
    Matrix loss{{0, -1, 2, -6, 102}};
    CheckCloseMatrix(relu_->BackPropagation(loss), Matrix{{0, 0, 0, -6, 102}});
}

class LeakyReLUTest : public testing::Test {
protected:
    Activation leaky_relu_ = neural_net::LeakyReLU();
    Matrix input_ = Matrix{{-10, -5, 0, 5, 10}};
};

TEST_F(LeakyReLUTest, Apply) {
    CheckCloseMatrix(leaky_relu_->Apply(input_), Matrix{{-0.1, -0.05, 0, 5, 10}});
}

TEST_F(LeakyReLUTest, BackPropagation) {
    leaky_relu_->Apply(input_.transpose());
    Matrix loss{{0, -1, 2, -6, 102}};
    CheckCloseMatrix(leaky_relu_->BackPropagation(loss), Matrix{{0, -0.01, 0, -6, 102}});
}

class SigmoidTest : public testing::Test {
protected:
    Activation sigmoid_ = neural_net::Sigmoid();
    Matrix input_ = Matrix{{-2, -1, 0, 0.5, 1, 5}};
};

TEST_F(SigmoidTest, Apply) {
    CheckCloseMatrix(sigmoid_->Apply(input_),
                     Matrix{{0.11920292202204383, 0.26894142136992605, 0.5, 0.6224593312018958,
                             0.731058578630074, 0.9933071490757268}});
}

TEST_F(SigmoidTest, BackPropagation) {
    sigmoid_->Apply(input_.transpose());
    Matrix loss{{-1, -0.5, 0, 0.3, 0.5, 1}};
    CheckCloseMatrix(sigmoid_->BackPropagation(loss),
                     Matrix{{-0.104993582, -0.098306, 0, 0.0705011, 0.098306, 0.00664803}});
}

class SoftmaxTest : public testing::Test {
protected:
    Activation softmax_ = neural_net::Softmax();
    Matrix input_ = Matrix{{1, 2, 1}, {1, 2, 1}};
};

TEST_F(SoftmaxTest, Correctness) {
    Matrix output = softmax_->Apply(input_.transpose());
    CheckCloseMatrix(output.transpose(), Matrix{{0.21194157, 0.5761169, 0.21194157},
                                                {0.21194157, 0.5761169, 0.21194157}});
}

TEST_F(SoftmaxTest, BackPropagation) {
    softmax_->Apply(input_.transpose());
    Matrix loss{{0.2, 2, 0.5}, {-0.2, -0.3, 0}};
    CheckCloseMatrix(
        softmax_->BackPropagation(loss),
        Matrix{{-0.233261347, 0.402940273, -0.169678867}, {0.00322647, -0.0488413, 0.0456148}});
}

TEST_F(SoftmaxTest, BigInput) {
    Matrix input{{1000, 2000, 3000}};
    Matrix output = softmax_->Apply(input.transpose());
    ASSERT_TRUE(output.array().allFinite());
}

class TanhTest : public testing::Test {
protected:
    Activation tanh_ = neural_net::Tanh();
    Matrix input_ = Matrix{{-2, -1.3, -0.5, 0, 0.2, 0.5, 1}};
};

TEST_F(TanhTest, Apply) {
    CheckCloseMatrix(tanh_->Apply(input_), Matrix{{-0.964027584, -0.861723125, -0.462117136, 0,
                                                   0.197375298, 0.462117136, 0.761594}});
}

TEST_F(TanhTest, BackPropogation) {
    tanh_->Apply(input_.transpose());
    Matrix loss{{-0.3, -1, 0.6, 2, 1.4, -1.3, 0.4}};
    CheckCloseMatrix(tanh_->BackPropagation(loss), Matrix{{-0.0211952, -0.257433236, 0.471868664, 2,
                                                           1.34546018, -1.02238202, 0.167989805}});
}
