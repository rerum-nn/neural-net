#include "Adam.h"

namespace neural_net {

Adam::Adam(double lr, double beta_1, double beta_2, FastStart is_fast_start)
    : learning_rate_(lr),
      beta_1_(beta_1),
      beta_2_(beta_2),
      is_fast_start_(is_fast_start == FastStart::Enable),
      cur_beta_1_(beta_1),
      cur_beta_2_(beta_2) {
    assert(learning_rate_ > 0);
    assert(0 <= beta_1 && beta_1 < 1);
    assert(0 <= beta_2 && beta_2 < 1);
}

void Adam::InitParameters(const std::vector<Linear>& layers) {
    momentums_.clear();
    velocities_.clear();

    momentums_.resize(layers.size());
    velocities_.resize(layers.size());
    if (!is_fast_start_) {
        cur_beta_1_ = 0;
        cur_beta_2_ = 0;
    } else {
        cur_beta_1_ = beta_1_;
        cur_beta_2_ = beta_2_;
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        momentums_[i].weights_grad.resizeLike(layers[i].GetWeights());
        momentums_[i].bias_grad.resizeLike(layers[i].GetBias());
        momentums_[i].weights_grad.setZero();
        momentums_[i].bias_grad.setZero();

        velocities_[i].weights_grad.resizeLike(layers[i].GetWeights());
        velocities_[i].bias_grad.resizeLike(layers[i].GetBias());
        velocities_[i].weights_grad.setZero();
        velocities_[i].bias_grad.setZero();
    }
}

void Adam::Update(const UpdatePack& pack, size_t layer_id) {
    assert(layer_id < momentums_.size());
    GradsPack& momentum = momentums_[layer_id];
    GradsPack& velocity = velocities_[layer_id];

    assert(pack.weights.rows() == pack.weights_grad.rows() &&
           pack.weights.cols() == pack.weights_grad.cols());
    assert(pack.bias.size() == pack.bias_grad.size());
    assert(momentum.weights_grad.cols() == pack.weights_grad.cols() &&
           momentum.weights_grad.rows() == pack.weights_grad.rows());
    assert(velocity.weights_grad.cols() == pack.weights_grad.cols() &&
           velocity.weights_grad.rows() == pack.weights_grad.rows());
    assert(momentum.bias_grad.size() == pack.bias_grad.size());
    assert(velocity.bias_grad.size() == pack.bias_grad.size());

    momentum.weights_grad = beta_1_ * momentum.weights_grad + (1 - beta_1_) * pack.weights_grad;
    velocity.weights_grad = beta_2_ * velocity.weights_grad +
                            (1 - beta_2_) * pack.weights_grad.cwiseProduct(pack.weights_grad);
    momentum.bias_grad = beta_1_ * momentum.bias_grad + (1 - beta_1_) * pack.bias_grad;
    velocity.bias_grad =
        beta_2_ * velocity.bias_grad + (1 - beta_2_) * pack.bias_grad.cwiseProduct(pack.bias_grad);

    pack.weights -=
        learning_rate_ * ((momentum.weights_grad / (1 - cur_beta_1_)).array() /
                          (((velocity.weights_grad / (1 - cur_beta_2_)).array() + kEpsilon).sqrt()))
                             .matrix();
    pack.bias -=
        learning_rate_ * ((momentum.bias_grad / (1 - cur_beta_1_)).array() /
                          (((velocity.bias_grad / (1 - cur_beta_2_)).array() + kEpsilon).sqrt()))
                             .matrix();
}

void Adam::BatchCallback() {
    cur_beta_1_ *= beta_1_;
    cur_beta_2_ *= beta_2_;
}

void Adam::EpochCallback(size_t epoch, size_t max_epoch) {
}

}  // namespace neural_net
