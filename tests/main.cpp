#include "../src/ActivationFunctions/Sigmoid.h"
#include "../src/Network.h"
#include "../src/Types.h"

#include <iostream>
#include <gtest/gtest.h>

using neural_net::Network;
using neural_net::Sigmoid;
using neural_net::Vector;

TEST(GeneralWork, NoThrowPredict) {
    Network network({4, 9, 5}, {Sigmoid(), Sigmoid()});
    Vector vector(4);
    vector << 4, 5, 3, 1;
    Vector res;
    ASSERT_NO_THROW(res = network.Predict(vector));
    std::cout << "res:" << std::endl;
    for (size_t i = 0; i < res.size(); ++i) {
        std::cout << i << ": " << res[i] << std::endl;
    }
}

int main() {
  testing::InitGoogleTest();

  return RUN_ALL_TESTS();
}
