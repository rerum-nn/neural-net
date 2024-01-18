#include "Network.h"

#include <cassert>
#include <stdexcept>

namespace neural_net {

Network::Network(std::initializer_list<size_t> layer_sizes,
                 std::initializer_list<ActivationFunction> functions) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument(
            "there should be at least two layers in a network: input and output");
    }
    if (layer_sizes.size() - 1 != functions.size()) {
        throw std::invalid_argument(
            "sizes of list of sizes and list of activation functions are not equal");
    }

    layers_.reserve(layer_sizes.size() - 1);
    auto input_size_it = layer_sizes.begin();
    auto output_size_it = input_size_it;
    auto activation_function_it = functions.begin();
    ++output_size_it;

    while (output_size_it != layer_sizes.end()) {
        layers_.emplace_back(*input_size_it, *output_size_it, *activation_function_it);
        ++input_size_it;
        ++output_size_it;
        ++activation_function_it;
    }
}
Vector Network::Predict(const Vector& input_data) {
    Vector res(input_data);

    for (Layer& layer : layers_) {
        layer.NormalRandomInit();
        layer.Forward(&res);
    }

    return res;
}
Vector Network::Predict(const std::initializer_list<double>& input_data) {
    Vector res({input_data});

    for (Layer& layer : layers_) {
        layer.Forward(&res);
    }

    return res;
}

}  // namespace neural_net
