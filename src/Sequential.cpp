#include "Sequential.h"

#include <cassert>
#include <string>

namespace neural_net {

Sequential::Sequential(std::initializer_list<Layer> layers) : layers_(layers) {
    assert(layers.size() >= 1 && "there should be at least one layer in a network");
}

Vector Sequential::Predict(const Vector& input_vector) {
    Vector iteration = input_vector;

    for (Layer& layer : layers_) {
        iteration = layer->Apply(iteration);
    }

    return iteration;
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

}  // namespace neural_net
