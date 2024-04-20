#pragma once

#include "Layers/Layer.h"
#include "Types.h"

#include <string>
#include <vector>

namespace neural_net {

class Sequential {
public:
    Sequential(std::initializer_list<Layer> layers);

    Matrix Predict(const Matrix& input_data);

    Sequential& AddLayer(const Layer& layer);
    Sequential& AddLayer(Layer&& layer);

    std::vector<Layer>& GetLayers();

private:
    std::vector<Layer> layers_;
};

}  // namespace neural_net
