#include "SGD.h"

#include <iostream>
#include <ranges>

namespace neural_net {
SGD::SGD(double lr, double momentum) : learning_rate_(lr), moment_(momentum) {
}

void SGD::operator()(Sequential& sequential, const Matrix& input_data, const Matrix& labels,
                     const LossFunction& loss, size_t max_epoch) const {
    std::vector<Layer>& layers = sequential.GetLayers();

    std::vector<std::vector<Matrix>> old_grad(layers.size());

    for (size_t epoch = 1; epoch <= max_epoch; ++epoch) {
        Matrix label = labels;
        Matrix output = sequential.Predict(input_data);

        Matrix nabla = loss->LossGradient(output, label);
        for (size_t i = 0; i < layers.size(); ++i) {
            size_t pos = layers.size() - 1 - i;
            Layer& layer = layers[pos];
            UpdateParameter(layer->GetGradients(nabla), old_grad[pos]);
            nabla = layer->BackPropagation(nabla);
        }
        std::cout << "\rEpoch [" << epoch << "/" << max_epoch
                  << "] Error: " << loss->Loss(sequential.Predict(input_data), labels);
    }
}

void SGD::UpdateParameter(const std::vector<ParametersGrad>& pack,
                          std::vector<Matrix>& old_grad) const {
    if (old_grad.empty()) {
        for (size_t i = 0; i < pack.size(); ++i) {
            const ParametersGrad& param = pack[i];
            Matrix delta = param.grad * learning_rate_;
            param.param -= delta;
            old_grad.push_back(delta);
        }
        return;
    }

    assert(pack.size() == old_grad.size());
    for (size_t i = 0; i < pack.size(); ++i) {
        const ParametersGrad& param = pack[i];
        Matrix delta = old_grad[i] * moment_ + param.grad * learning_rate_;
        param.param -= delta;
        old_grad[i] = delta;
    }
}

}  // namespace neural_net