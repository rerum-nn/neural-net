#include "../src/Layers/Linear.h"
#include "../src/Layers/Sigmoid.h"
#include "../src/Layers/Softmax.h"
#include "../src/LossFunctions/BinaryCrossEntropy.h"
#include "../src/Optimizers/Optimizers.h"
#include "../src/Sequential.h"
#include "../src/Types.h"

#include <gtest/gtest.h>

#include <iostream>

using namespace neural_net;

TEST(Models, XOR) {
    Sequential network({Linear(2, 2), Sigmoid(), Linear(2, 1), Sigmoid()});
    Matrix train_data{{0, 1, 0, 1}, {0, 0, 1, 1}};
    Matrix labels{{0, 1, 1, 0}};

    Optimizer adam = Optimizers::Adam(0.3);
    adam(network, train_data, labels, BinaryCrossEntropy(), 1000);
    for (Index i = 0; i < 4; ++i) {
        Vector vector = train_data.col(i);
        Vector res;
        ASSERT_NO_THROW(res = network.Predict(vector));
        std::cout << "res:" << std::endl;
        for (size_t j = 0; j < res.size(); ++j) {
            std::cout << j << ": " << res[j] << std::endl;
        }
    }

}

TEST(CheckLayers, Softmax) {
    Sequential network({Softmax()});
    Vector data{{1000, 2000, 3000}};
    Vector ans = network.Predict(data);
    for (Index i = 0; i < ans.rows(); ++i) {
        std::cout << i << ": " << ans[i] << std::endl;
    }
}
