#include "../src/Layers/Sigmoid.h"
#include "../src/Network.h"
#include "../src/LossFunctions/BinaryCrossEntropy.h"
#include "../src/Types.h"
#include "../src/Layers/Linear.h"

#include <iostream>
#include <gtest/gtest.h>

using namespace neural_net;

TEST(GeneralWork, NoThrowPredict) {
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

int main() {
  testing::InitGoogleTest();

  return RUN_ALL_TESTS();
}
