#include "Sequential.h"

#include "BatchSlicer.h"

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

    for (size_t epoch = 1; epoch <= max_epoch; ++epoch) {
        for (const auto& [batch_data, batch_labels] :
             BatchSlicer(&input_data, &labels, batch_size)) {
            Matrix output = Predict(batch_data);

            Matrix nabla = loss->LossGradient(output, batch_labels);
            for (size_t i = 0; i < layers_.size(); ++i) {
                size_t pos = layers_.size() - 1 - i;
                Layer& layer = layers_[pos];
                optimizer->Update(layer->GetGradients(nabla), pos);
                nabla = layer->BackPropagation(nabla);
            }
            optimizer->BatchCallback();
        }
        double loss_value = loss->Loss(Predict(input_data), labels);
        optimizer->EpochCallback(epoch, max_epoch, loss_value);
        err.push_back(loss_value);
    }

    return err;
}

}  // namespace neural_net
