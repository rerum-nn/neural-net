#include "Sequential.h"

#include "Utils/BatchSlicer.h"
#include "Utils/DataManipulate.h"

#include <cassert>
#include <iostream>
#include <string>

namespace neural_net {

Sequential::Sequential(std::initializer_list<Layer> layers) : layers_(layers) {
    assert(layers.size() >= 1 && "there should be at least one layer in a network");
}

Matrix Sequential::Predict(const Matrix& input_data) {
    Matrix iteration = input_data.transpose();

    for (Layer& layer : layers_) {
        iteration = layer->Apply(iteration);
    }

    return iteration.transpose();
}

std::vector<double> Sequential::Evaluate(const Matrix& input_data, const Matrix& labels,
                                         const LossFunction& loss,
                                         std::initializer_list<Metric> metrics) {
    std::vector<double> values;
    Matrix output = Predict(input_data);
    values.push_back(loss->Loss(output, labels));
    for (const Metric& metric : metrics) {
        values.push_back(metric(output, labels));
    }
    return values;
}

Sequential& Sequential::AddLayer(const Layer& layer) {
    layers_.push_back(layer);
    return *this;
}

Sequential& Sequential::AddLayer(Layer&& layer) {
    layers_.push_back(std::move(layer));
    return *this;
}

std::vector<Layer>& Sequential::GetLayers() {
    return layers_;
}

std::vector<double> Sequential::Fit(const Matrix& input_data, const Matrix& labels,
                                    FitParameters fit_parameters) {
    LossFunction& loss = fit_parameters.loss;
    Optimizer& optimizer = fit_parameters.optimizer;
    size_t max_epoch = fit_parameters.max_epoch;
    size_t batch_size = fit_parameters.batch_size;

    std::vector<double> err;
    err.reserve(fit_parameters.max_epoch);

    fit_parameters.optimizer->InitParameters(layers_);

    for (size_t epoch = 1; epoch <= max_epoch; ++epoch) {
        auto [train_data, train_labels, validate_data, validate_labels] = TrainTestSplit(
            input_data, labels, (1 - fit_parameters.validate_ratio), ShuffleMode::Shuffle);
        time_t start = time(nullptr);
        for (const auto& [batch_data, batch_labels] :
             BatchSlicer(train_data, train_labels, batch_size, ShuffleMode::Static)) {
            Matrix output = Predict(batch_data);
            Matrix nabla = loss->LossGradient(output, batch_labels);
            std::vector<std::vector<ParametersGrad>> grads;
            grads.reserve(layers_.size());
            for (size_t i = 0; i < layers_.size(); ++i) {
                size_t pos = layers_.size() - 1 - i;
                Layer& layer = layers_[pos];
                Matrix next_nabla = layer->BackPropagation(nabla);
                grads.push_back(layer->GetGradients(nabla));
                nabla = next_nabla;
            }
#pragma omp parallel for
            for (size_t i = 0; i < grads.size(); ++i) {
                size_t pos = layers_.size() - 1 - i;
                optimizer->Update(grads[pos], pos);
            }
            optimizer->BatchCallback();
        }
        time_t end = time(nullptr);
        Matrix validate_output = Predict(validate_data);
        double loss_value = loss->Loss(validate_output, validate_labels);
        double acc = Metric::CategoricalAccuracy()(validate_output, validate_labels);
        optimizer->EpochCallback(epoch, max_epoch);
        std::cout << "Time: " << end - start << " Acc: " << acc << std::endl;
        err.push_back(loss_value);
    }

    return err;
}

}  // namespace neural_net
