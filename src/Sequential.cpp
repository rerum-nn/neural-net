#include "Sequential.h"

#include "Utils/BatchSlicer.h"

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

    fit_parameters.optimizer->InitParameters(layers_);

    BatchSlicer batch_slicer(input_data, labels, batch_size);
    for (size_t epoch = 1; epoch <= max_epoch; ++epoch) {
        double total_loss = 0;
        int count_batch = 0;
        time_t start = time(nullptr);
        for (const auto& [batch_data, batch_labels] : batch_slicer) {
            Matrix output = Predict(batch_data);

            double loss_value = loss->Loss(output, batch_labels);
            total_loss += loss_value;
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
            for (size_t i = 0; i < grads.size(); ++i) {
                size_t pos = layers_.size() - 1 - i;
                optimizer->Update(grads[pos], pos);
            }
            optimizer->BatchCallback();
            ++count_batch;
        }
        time_t end = time(nullptr);
        total_loss /= count_batch;
        optimizer->EpochCallback(epoch, max_epoch, total_loss);
        std::cout << "Time: " << end - start << std::endl;
        batch_slicer.Shuffle();
        err.push_back(total_loss);
    }

    return err;
}

}  // namespace neural_net
