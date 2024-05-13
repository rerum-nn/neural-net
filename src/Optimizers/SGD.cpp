#include "SGD.h"

namespace neural_net {

SGD::SGD(double lr, double momentum) : learning_rate_(lr), moment_(momentum) {
    assert(learning_rate_ > 0);
    assert(moment_ >= 0);
}

void SGD::InitParameters(const std::vector<Linear>& layers) {
    old_grads_.clear();
    old_grads_.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        old_grads_[i].weights_grad.resizeLike(layers[i].GetWeights());
        old_grads_[i].bias_grad.resizeLike(layers[i].GetBias());
        old_grads_[i].weights_grad.setZero();
        old_grads_[i].bias_grad.setZero();
    }
}

void SGD::Update(const UpdatePack& pack, size_t layer_id) {
    assert(layer_id < old_grads_.size());
    GradsPack& old_grad = old_grads_[layer_id];

    assert(pack.weights.rows() == pack.weights_grad.rows() &&
           pack.weights.cols() == pack.weights_grad.cols());
    assert(pack.bias.size() == pack.bias_grad.size());
    assert(old_grad.weights_grad.cols() == pack.weights_grad.cols() &&
           old_grad.weights_grad.rows() == pack.weights_grad.rows());
    assert(old_grad.bias_grad.size() == pack.bias_grad.size());

    old_grad.weights_grad = old_grad.weights_grad * moment_ + pack.weights_grad * learning_rate_;
    old_grad.bias_grad = old_grad.bias_grad * moment_ + pack.bias_grad * learning_rate_;

    pack.weights -= old_grad.weights_grad;
    pack.bias -= old_grad.bias_grad;
}

void SGD::BatchCallback() {
}

void SGD::EpochCallback(size_t epoch, size_t max_epoch) {
}

}  // namespace neural_net
