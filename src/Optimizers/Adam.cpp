#include "Adam.h"

#include <iostream>

namespace neural_net {
Adam::Adam(double lr, double beta_1, double beta_2, FastStart is_fast_start)
    : learning_rate_(lr),
      beta_1_(beta_1),
      beta_2_(beta_2),
      is_fast_start_(is_fast_start == FastStart::Enable),
      cur_beta_1_(beta_1),
      cur_beta_2_(beta_2) {
}

void Adam::InitParameters(const std::vector<Layer>& layers) {
    first_moments_.clear();
    second_moments_.clear();

    first_moments_.resize(layers.size());
    second_moments_.resize(layers.size());
    if (is_fast_start_) {
        cur_beta_1_ = 0;
        cur_beta_2_ = 0;
    } else {
        cur_beta_1_ = beta_1_;
        cur_beta_2_ = beta_2_;
    }
}

void Adam::Update(const std::vector<ParametersGrad>& pack, size_t layer_id) {
    std::vector<Matrix>& first_moment = first_moments_[layer_id];
    std::vector<Matrix>& second_moment = second_moments_[layer_id];
    if (first_moment.empty() && second_moment.empty()) {
        first_moment.resize(pack.size());
        second_moment.resize(pack.size());
        for (size_t i = 0; i < pack.size(); ++i) {
            const ParametersGrad& param = pack[i];
            first_moment[i] = (1 - beta_1_) * param.grad;
            second_moment[i] = (1 - beta_2_) * param.grad.cwiseProduct(param.grad);
            param.param -= learning_rate_ *
                           ((first_moment[i] / (1 - cur_beta_1_)).array() /
                            (((second_moment[i] / (1 - cur_beta_2_)).array() + kEpsilon).sqrt()))
                               .matrix();
        }
        return;
    }

    assert(pack.size() == first_moment.size() && pack.size() == second_moment.size());
    for (size_t i = 0; i < pack.size(); ++i) {
        const ParametersGrad& param = pack[i];
        first_moment[i] = beta_1_ * first_moment[i] + (1 - beta_1_) * param.grad;
        second_moment[i] = beta_2_ * second_moment[i] + (1 - beta_2_) * param.grad.cwiseProduct(param.grad);
        param.param -=
            learning_rate_ * ((first_moment[i] / (1 - cur_beta_1_)).array() /
                              (((second_moment[i] / (1 - cur_beta_2_)).array() + kEpsilon).sqrt()))
                                 .matrix();
    }
}

void Adam::BatchCallback() {
    cur_beta_1_ *= beta_1_;
    cur_beta_2_ *= beta_2_;
}

void Adam::EpochCallback(size_t epoch, size_t max_epoch, double loss) {
    std::cout << "Epoch [" << epoch << "/" << max_epoch << "] Error: " << loss << std::endl;
}
}  // namespace neural_net