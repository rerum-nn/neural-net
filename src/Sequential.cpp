#include "Sequential.h"

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
        //        for (Index i = 0; i < iteration.rows(); ++i) {
        //            for (Index j = 0; j < iteration.cols(); ++j) {
        //                std::cout << iteration(i, j) << ' ' << std::endl;
        //            }
        //            std::cout << std::endl;
        //        }
        //        std::cout << std::endl;
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

}  // namespace neural_net
