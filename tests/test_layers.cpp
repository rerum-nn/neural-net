#include "Layers/Layer.h"
#include "Layers/Linear.h"
#include "Layers/ReLU.h"
#include "Layers/Sigmoid.h"
#include "Layers/Softmax.h"
#include "TestHelpers.h"
#include "Types.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

using neural_net::Index;
using neural_net::Layer;
using neural_net::Linear;
using neural_net::Matrix;
using neural_net::ReLU;
using neural_net::Sigmoid;
using neural_net::Softmax;
using neural_net::Vector;
using testing::DoubleNear;

TEST(Linear, Correctness) {
    Layer linear = Linear(Matrix{{1.0, 3.0}, {-2.0, -4.0}}, Vector{{5.0, -6.0}});
    Matrix input{{-1.0, 2.0}};
    input = linear->Apply(input.transpose());
    Layer relu = ReLU();
    input = relu->Apply(input);
    CheckCloseMatrix(input.transpose(), Matrix{{10.0, 0.0}});

    linear = Linear(Matrix{{1.0, 3.0}, {-2.0, -4.0}}, Vector{{0., 0.}});
    input = Matrix{{-1.0, 2.0}}.transpose();
    input = linear->Apply(input);
    CheckCloseMatrix(input.transpose(), Matrix{{5.0, -6.0}});
}

TEST(ReLU, Correctness) {
    Layer relu = ReLU();
    Matrix input{{-10, -5, 0, 5, 10}};
    input = relu->Apply(input.transpose());
    CheckCloseMatrix(input.transpose(), Matrix{{0, 0, 0, 5, 10}});
}

TEST(Softmax, Correctness) {
    Layer softmax = Softmax();
    Matrix input{{1, 2, 1}, {1, 2, 1}};
    input = softmax->Apply(input.transpose());
    CheckCloseMatrix(
        input.transpose(), Matrix{{0.21194157, 0.5761169, 0.21194157}, {0.21194157, 0.5761169, 0.21194157}});
}
