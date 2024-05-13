#include "Datasets/MNIST/MnistDataset.h"
#include "LossFunctions/CategoricalCrossEntropy.h"
#include "Metrics/Metric.h"
#include "Sequential.h"
#include "Types.h"
#include "Utils/DataManipulate.h"

#include <fstream>
#include <iostream>

using namespace neural_net;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "./test_deserialize /path/to/model.fnn\n";
        return 1;
    }
    Sequential sequential;
    std::ifstream model_file(argv[1]);
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

