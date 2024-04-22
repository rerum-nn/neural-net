#include "Layers/Sigmoid.h"
#include "Optimizers/Optimizer.h"
#include "TestHelpers.h"
#include "Types.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

using neural_net::Index;
using neural_net::Matrix;
using neural_net::Optimizer;
using neural_net::ParametersGrad;
using neural_net::Sigmoid;
using testing::DoubleNear;

TEST(SGD, SingleStep) {
    Optimizer sgd = Optimizer::SGD(0.5);
    Matrix vars{{1.0, 2.0, 3.0, 4.0}};
    ParametersGrad grads{vars, Matrix{{1.0, 6.0, 7.0, 2.0}}};
    sgd->InitParameters({Sigmoid()});
    sgd->Update({grads}, 0);
    CheckCloseMatrix(vars, Matrix{{0.5, -1.0, -0.5, 3.0}});
}

TEST(SGD, EmptyGrad) {
    Optimizer sgd = Optimizer::SGD(0.5);
    Matrix vars{{1.0, 2.0, 3.0, 4.0}};
    ParametersGrad grads{vars, Matrix{}};
    sgd->InitParameters({Sigmoid()});
    ASSERT_DEATH(sgd->Update({grads}, 0), "");
}

TEST(Adam, SingleStep) {
    Optimizer adam = Optimizer::Adam(0.5);
    Matrix vars{{1.0, 2.0, 3.0, 4.0}};
    ParametersGrad grads{vars, Matrix{{1.0, 6.0, 7.0, 2.0}}};
    adam->InitParameters({Sigmoid()});
    adam->Update({grads}, 0);
    CheckCloseMatrix(vars, Matrix{{0.5, 1.5, 2.5, 3.5}});
}

TEST(RMSProp, SingleStep) {
    Optimizer rmsprop = Optimizer::RMSProp(0.5);
    Matrix vars{{1.0, 2.0, 3.0, 4.0}};
    ParametersGrad grads{vars, Matrix{{1.0, 6.0, 7.0, 2.0}}};
    rmsprop->InitParameters({Sigmoid()});
    rmsprop->Update({grads}, 0);
    CheckCloseMatrix(vars, Matrix{{-0.5811, 0.4189, 1.4189, 2.4189}}, 1e-4);
}
