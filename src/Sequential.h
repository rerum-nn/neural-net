#pragma once

#include "Layers/Layer.h"
#include "Types.h"

#include <string>
#include <vector>

namespace neural_net {

class Sequential {
public:
    Sequential(std::initializer_list<Layer> layers);

    Vector Predict(const Vector& input_vector);

    Sequential& AddLayer(const Layer& layer);
    Sequential& AddLayer(Layer&& layer);

    std::vector<Layer>& GetLayers();

private:
    std::vector<Layer> layers_;
};

}  // namespace neural_net
