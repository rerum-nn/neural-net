#include "Optimizers/Optimizer.h"
#include "TestHelpers.h"
#include "Types.h"

#include <gtest/gtest.h>

using neural_net::Index;
using neural_net::Matrix;
using neural_net::Vector;
using neural_net::Optimizer;
using neural_net::UpdatePack;
using neural_net::Linear;

TEST(SGD, SingleStep) {
    Optimizer sgd = Optimizer::SGD(0.5);
    Matrix vars{{1.0, 2.0, 3.0, 4.0}};
    Vector bias{{0.0}};
    UpdatePack grads{vars, Matrix{{1.0, 6.0, 7.0, 2.0}}, bias, Vector{{0}}};
    sgd->InitParameters({Linear(4, 1)});
    sgd->Update({grads}, 0);
    CheckCloseMatrix(vars, Matrix{{0.5, -1.0, -0.5, 3.0}});
}

TEST(Adam, SingleStep) {
    Optimizer adam = Optimizer::Adam(0.5);
    Matrix vars{{1.0, 2.0, 3.0, 4.0}};
    Vector bias{{0}};
    UpdatePack grads{vars, Matrix{{1.0, 6.0, 7.0, 2.0}}, bias, Vector{{0}}};
    adam->InitParameters({Linear(4, 1)});
    adam->Update({grads}, 0);
    CheckCloseMatrix(vars, Matrix{{0.5, 1.5, 2.5, 3.5}});
}

TEST(RMSProp, SingleStep) {
    Optimizer rmsprop = Optimizer::RMSprop(0.5);
    Matrix vars{{1.0, 2.0, 3.0, 4.0}};
    Vector bias{{0}};
    UpdatePack grads{vars, Matrix{{1.0, 6.0, 7.0, 2.0}}, bias, Vector{{0}}};
    rmsprop->InitParameters({Linear(4, 1)});
    rmsprop->Update({grads}, 0);
    CheckCloseMatrix(vars, Matrix{{-0.5811, 0.4189, 1.4189, 2.4189}}, 1e-4);
}
