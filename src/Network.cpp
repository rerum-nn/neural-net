#include "Network.h"

#include <cassert>
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

Network& Network::AddLayer(const Layer& layer) {
    layers_.push_back(layer);
    return *this;
}

Network& Network::AddLayer(Layer&& layer) {
    layers_.push_back(std::move(layer));
    return *this;
}

std::vector<Layer>& Network::GetLayers() {
    return layers_;
}

}  // namespace neural_net
