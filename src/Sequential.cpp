#include "Sequential.h"

#include "Utils/BatchSlicer.h"
#include "Utils/DataManipulate.h"
#include "Utils/Timer.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>

namespace neural_net {

Sequential::Sequential(std::initializer_list<Linear> layers) : layers_(layers) {
}

Matrix Sequential::Predict(const Matrix& input_data) {
    Matrix iteration = input_data.transpose();

    for (Linear& layer : layers_) {
        iteration = layer.Apply(iteration);
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

Sequential& Sequential::AddLayer(const Linear& layer) {
    layers_.push_back(layer);
    return *this;
}

Sequential& Sequential::AddLayer(Linear&& layer) {
    layers_.push_back(std::move(layer));
    return *this;
}

std::vector<Linear>& Sequential::GetLayers() {
    return layers_;
}

std::vector<double> Sequential::Fit(const Matrix& input_data, const Matrix& labels,
                                    FitParameters fit_parameters) {
    LossFunction& loss = fit_parameters.loss;
    Optimizer& optimizer = fit_parameters.optimizer;
    const std::vector<Metric>& metrics = fit_parameters.metrics;
    size_t max_epoch = fit_parameters.max_epoch;
    size_t batch_size = fit_parameters.batch_size;
    float validate_ratio = fit_parameters.validate_ratio;

    std::vector<double> loss_values;
    loss_values.reserve(max_epoch);

    optimizer->InitParameters(layers_);

    Timer timer;

    auto [train_data, train_labels, validate_data, validate_labels] =
        TrainTestSplit(input_data, labels, (1 - validate_ratio), ShuffleMode::Static);
    for (size_t epoch = 1; epoch <= max_epoch; ++epoch) {
        timer.Reset();
        for (const auto& [batch_data, batch_labels] :
             BatchSlicer(train_data, train_labels, batch_size, ShuffleMode::Static)) {
            Matrix output = Predict(batch_data);
            Matrix nabla = loss->LossGradient(output, batch_labels);
            std::vector<UpdatePack> grads;
            grads.reserve(layers_.size());
            for (size_t i = 0; i < layers_.size(); ++i) {
                size_t pos = layers_.size() - 1 - i;
                Linear& layer = layers_[pos];
                nabla = layer.BackPropagationActivation(nabla);
                Matrix next_nabla = layer.BackPropagation(nabla);
                grads.push_back(layer.GetGradients(nabla));
                nabla = next_nabla;
            }
#pragma omp parallel for
            for (size_t i = 0; i < grads.size(); ++i) {
                size_t pos = layers_.size() - 1 - i;
                optimizer->Update(grads[pos], i);
            }
            optimizer->BatchCallback();
        }
        optimizer->EpochCallback(epoch, max_epoch);
        std::cout << "Epoch [" << epoch << "/" << max_epoch
                  << "] Time: " << timer.GetTimerString() << '\n';
        if (validate_data.size() > 0) {
            Matrix validate_output = Predict(validate_data);
            double loss_value = loss->Loss(validate_output, validate_labels);
            loss_values.push_back(loss_value);

            std::cout << "loss: " << loss_value;
            for (const Metric& metric : metrics) {
                std::cout << ' ' << metric.GetName() << ": "
                          << metric(validate_output, validate_labels);
            }
        }
        std::cout << "\n\n";
    }

    return loss_values;
}

void Sequential::Serialize(std::ostream& os) const {
    os << layers_.size();
    for (const Linear& layer : layers_) {
        layer.Serialize(os);
    }
}

void Sequential::Deserialize(std::istream& is) {
    size_t size;
    is >> size;
    layers_.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        layers_.push_back(DeserializeLayer(is));
    }
}

}  // namespace neural_net
