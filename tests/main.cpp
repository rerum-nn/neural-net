#include "../src/Layers/Linear.h"
#include "../src/Layers/Sigmoid.h"
#include "../src/Layers/Softmax.h"
#include "../src/LossFunctions/BinaryCrossEntropy.h"
#include "../src/Network.h"
#include "../src/Types.h"

#include <gtest/gtest.h>

#include <iostream>

using namespace neural_net;

TEST(Models, XOR) {
    Network network({Linear(2, 2), Sigmoid(), Linear(2, 1), Sigmoid()});
    Matrix train_data{
        {0, 1, 0, 1},
        {0, 0, 1, 1}
    };
    Matrix expected_answers{
        {0, 1, 1, 0}
    };
    for (size_t i = 1; i < 100000;++ i) {
        network.Fit(train_data, expected_answers, BinaryCrossEntropy());
    }
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
    Network network({Softmax()});
    Vector data{
        {1000, 2000, 3000}
    };
    Vector ans = network.Predict(data);
    for (Index i = 0; i < ans.rows(); ++i) {
        std::cout << i << ": " << ans[i] << std::endl;
    }
}
