#include "RMSProp.h"

namespace neural_net {
RMSProp::RMSProp(double lr, double rho) : learning_rate_(lr), rho_(rho) {
}

void RMSProp::operator()(Network& network, const Matrix& input_data, const Matrix& labels,
                         const LossFunction& loss, size_t max_epoch) const {
    std::vector<Layer>& layers = network.GetLayers();

    std::vector<std::vector<Matrix>> old_grad(layers.size());

    for (size_t epoch = 1; epoch <= max_epoch; ++epoch) {
        for (Index batch = 0; batch < input_data.cols(); ++batch) {
            Vector label = labels.col(batch);
            Vector output = network.Predict(input_data.col(batch));

            RowVector nabla = loss->LossGradient(output, label);
            for (size_t i = 0; i < layers.size(); ++i) {
                size_t pos = layers.size() - 1 - i;
                Layer& layer = layers[pos];
                UpdateParameter(layer->GetGradients(nabla), old_grad[pos]);
                nabla = layer->BackPropagation(nabla);
            }
        }
    }
}

void RMSProp::UpdateParameter(const std::vector<ParametersGrad>& pack,
                              std::vector<Matrix>& old_grad) const {
    if (old_grad.empty()) {
        for (size_t i = 0; i < pack.size(); ++i) {
            const ParametersGrad& param = pack[i];
            Matrix delta = (1 - rho_) * param.grad.cwiseProduct(param.grad);
            param.param -=
                learning_rate_ * (param.grad.array() / (delta.array().sqrt() + kEpsilon)).matrix();
            old_grad.push_back(delta);
        }
        return;
    }

    assert(pack.size() == old_grad.size());
    for (size_t i = 0; i < pack.size(); ++i) {
        const ParametersGrad& param = pack[i];
        Matrix delta = rho_ * old_grad[i] + (1 - rho_) * param.grad.cwiseProduct(param.grad);
        param.param -=
            learning_rate_ * (param.grad.array() / (delta.array().sqrt() + kEpsilon)).matrix();
        old_grad[i] = delta;
    }
}
}  // namespace neural_net