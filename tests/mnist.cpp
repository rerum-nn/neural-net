#include "Datasets/MNIST/MnistDataset.h"
#include "Layers/Linear.h"
#include "Layers/ReLU.h"
#include "Layers/Softmax.h"
#include "LossFunctions/CategoricalCrossEntropy.h"
#include "Optimizers/Optimizer.h"
#include "Sequential.h"
#include "Types.h"
#include "Utils/DataManipulate.h"

#include <iostream>

using namespace neural_net;

int main() {
    Eigen::setNbThreads(6);

    auto [x_train, y_train, x_test, y_test] = MnistDataset().LoadData();
    Matrix train_labels = IntLabelsToCategorical(y_train);
    Matrix test_labels = IntLabelsToCategorical(y_test);

    std::cout << "start of fitting\n";
    Sequential sequential({Linear(784, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, 10), Softmax()});
    sequential.Fit(x_train, train_labels, {CategoricalCrossEntropy(), Optimizer::Adam(), 10, 1});

}
