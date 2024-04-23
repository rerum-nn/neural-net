#include "SGD.h"

namespace neural_net {

SGD::SGD(double lr, double momentum) : learning_rate_(lr), moment_(momentum) {
}

void SGD::InitParameters(const std::vector<Linear>& layers) {
    old_grads_.clear();
    old_grads_.resize(layers.size());
}

void SGD::Update(const std::vector<ParametersGrad>& pack, size_t layer_id) {
    std::vector<Matrix>& old_grad = old_grads_[layer_id];
    if (old_grad.empty()) {
        for (size_t i = 0; i < pack.size(); ++i) {
            const ParametersGrad& param = pack[i];
            assert(param.param.rows() == param.grad.rows() &&
                   param.param.cols() == param.grad.cols());
            Matrix delta = param.grad * learning_rate_;
            param.param -= delta;
            old_grad.push_back(delta);
        }
        return;
    }

    assert(pack.size() == old_grad.size());
    for (size_t i = 0; i < pack.size(); ++i) {
        const ParametersGrad& param = pack[i];
        assert(param.param.rows() == param.grad.rows() && param.param.cols() == param.grad.cols());
        Matrix delta = old_grad[i] * moment_ + param.grad * learning_rate_;
        param.param -= delta;
        old_grad[i] = delta;
    }
}

void SGD::BatchCallback() {
}

void SGD::EpochCallback(size_t epoch, size_t max_epoch) {
}

}  // namespace neural_net
