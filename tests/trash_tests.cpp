#include "Layers/Linear.h"
#include "Layers/ReLU.h"
#include "Layers/Sigmoid.h"
#include "Layers/Softmax.h"
#include "LossFunctions/BinaryCrossEntropy.h"
#include "Optimizers/Optimizer.h"
#include "Sequential.h"
#include "Types.h"
#include "Datasets/MNIST/MnistDataset.h"
#include "Utils/DataManipulate.h"
#include "LossFunctions/CategoricalCrossEntropy.h"

#include <gtest/gtest.h>

#include <iostream>

using namespace neural_net;

TEST(Models, XOR) {
    Sequential sequential({Linear(2, 2), Sigmoid(), Linear(2, 1), Sigmoid()});
    Matrix train_data{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    Matrix labels{{0}, {1}, {1}, {0}};

    sequential.Fit(train_data, labels, {BinaryCrossEntropy(), Optimizer::SGD(0.3), 10000, 1});
    for (Index i = 0; i < 4; ++i) {
        RowVector vector = train_data.row(i);
        RowVector res;
        ASSERT_NO_THROW(res = sequential.Predict(vector));
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
