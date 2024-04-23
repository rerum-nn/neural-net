#include "Datasets/MNIST/MnistDataset.h"
#include "Layers/Activations/ReLU.h"
#include "Layers/Activations/Softmax.h"
#include "Layers/Linear.h"
#include "LossFunctions/CategoricalCrossEntropy.h"
#include "Metrics/Metric.h"
#include "Optimizers/Optimizer.h"
#include "Sequential.h"
#include "Types.h"
#include "Utils/DataManipulate.h"

#include <fstream>
#include <iostream>

using namespace neural_net;

int main() {
    Sequential sequential;
    std::ifstream model_file("model.fnn");
    sequential.Deserialize(model_file);

    auto [x_train, y_train, x_test, y_test] = MnistDataset().LoadData();
    Matrix train_labels = IntLabelsToCategorical(y_train);
    Matrix test_labels = IntLabelsToCategorical(y_test);

    auto test_metrics = sequential.Evaluate(x_test, test_labels, CategoricalCrossEntropy(),
                                            {Metric::CategoricalAccuracy()});
    std::cout << "Loss test: " << test_metrics[0] << " Acc: " << test_metrics[1] << '\n';

    size_t n = 1986;
    Matrix data = x_test.row(n);
    Matrix label = y_test.row(n);

    std::cout << "pred: " << sequential.Predict(data) << "  true:" << label << std::endl;

}

