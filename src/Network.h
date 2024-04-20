#pragma once

#include "Layers/Layer.h"
#include "Types.h"

#include <string>
#include <vector>

namespace neural_net {

class Network {
public:
    Network(std::initializer_list<Layer> layers);

    Vector Predict(const Vector& input_vector);

    Network& AddLayer(const Layer& layer);
    Network& AddLayer(Layer&& layer);

    std::vector<Layer>& GetLayers();

private:
    std::vector<Layer> layers_;
};

}  // namespace neural_net
