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
#include "Utils/Timer.h"

#include <fstream>
#include <iostream>

using neural_net::Linear;
using neural_net::Matrix;

int main() {
#ifdef _OPENMP
#ifndef THREADS
    omp_set_num_threads(omp_get_num_procs());
#else
    omp_set_num_threads(THREADS);
    std::cout << "Number of threads used: " << THREADS << std::endl;
#endif
#endif

    auto [x_train, y_train, x_test, y_test] = neural_net::MnistDataset().LoadData();
    Matrix train_labels = neural_net::IntLabelsToCategorical(y_train);
    Matrix test_labels = neural_net::IntLabelsToCategorical(y_test);

    std::cout << "start of fitting\n";
    neural_net::Timer timer;
    neural_net::Sequential sequential(
        {Linear(784, 512, neural_net::ReLU()), Linear(512, 10, neural_net::Softmax())});
    sequential.Fit(x_train, train_labels,
                   {.loss = neural_net::CategoricalCrossEntropy(),
                    .optimizer = neural_net::Optimizer::Adam(),
                    .max_epoch = 20,
                    .batch_size = 128,
                    .validate_ratio = 0.2,
                    .metrics = {neural_net::Metric::CategoricalAccuracy()}});

    std::cout << "Total_time: " << timer.GetTimerString() << std::endl;

    auto test_metrics =
        sequential.Evaluate(x_test, test_labels, neural_net::CategoricalCrossEntropy(),
                            {neural_net::Metric::CategoricalAccuracy()});
    std::cout << "Loss test: " << test_metrics[0] << " Acc: " << test_metrics[1] << '\n';

    std::ofstream model_file("model.fnn", std::ios::out);
    sequential.Serialize(model_file);
}
