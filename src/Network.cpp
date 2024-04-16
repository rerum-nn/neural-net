#include "Network.h"

#include <cassert>
#include <ranges>
#include <string>

namespace neural_net {

Network::Network(std::initializer_list<Layer> layers) : layers_(layers) {
    assert(layers.size() >= 1 && "there should be at least one layer in a network");
}

Vector Network::Predict(const Vector& input_vector) {
    Vector iteration = input_vector;

    for (Layer& layer : layers_) {
        iteration = layer->Apply(iteration);
    }

    return iteration;
}

void Network::Fit(const Matrix& input_data, const Matrix& labels, const LossFunction& loss,
                  Optimizer&& optimizer) {
    for (Index batch = 0; batch < input_data.cols(); ++batch) {
        Vector label = labels.col(batch);
        Vector output = Predict(input_data.col(batch));

        RowVector nabla = loss->LossGradient(output, label);
        for (Layer& layer : std::ranges::reverse_view(layers_)) {
            optimizer->Optimize(layer->GetGradients(nabla));
            nabla = layer->BackPropagation(nabla);
        }
    }
}

Network& Network::AddLayer(const Layer& layer) {
    layers_.push_back(layer);
    return *this;
}

Network& Network::AddLayer(Layer&& layer) {
    layers_.push_back(std::move(layer));
    return *this;
}

}  // namespace neural_net
