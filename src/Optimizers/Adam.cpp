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
        for (size_t i = 0; i < pack.size(); ++i) {
            const ParametersGrad& param = pack[i];
            Matrix m;
            Matrix v;
            m = (1 - beta_1_) * param.grad;
            v = (1 - beta_2_) * param.grad.cwiseProduct(param.grad);
            first_moment.push_back(m);
            second_moment.push_back(v);
            Matrix m_hat = m / (1 - cur_beta_1_);
            Matrix v_hat = v / (1 - cur_beta_2_);
            param.param -=
                learning_rate_ * (m_hat.array() / ((v_hat.array() + kEpsilon).sqrt())).matrix();
        }
        return;
    }

    assert(pack.size() == first_moment.size() && pack.size() == second_moment.size());
    for (size_t i = 0; i < pack.size(); ++i) {
        const ParametersGrad& param = pack[i];
        Matrix m;
        Matrix v;
        m = beta_1_ * first_moment[i] + (1 - beta_1_) * param.grad;
        v = beta_2_ * second_moment[i] + (1 - beta_2_) * param.grad.cwiseProduct(param.grad);
        first_moment[i] = m;
        second_moment[i] = v;
        Matrix m_hat = m / (1 - cur_beta_1_);
        Matrix v_hat = v / (1 - cur_beta_2_);
        param.param -=
            learning_rate_ * (m_hat.array() / ((v_hat.array() + kEpsilon).sqrt())).matrix();
    }
}

void Adam::BatchCallback() {
    cur_beta_1_ *= beta_1_;
    cur_beta_2_ *= beta_2_;
}

void Adam::EpochCallback(size_t epoch, size_t max_epoch, double loss) {
    std::cout << "\rEpoch [" << epoch << "/" << max_epoch << "] Error: " << loss;
}
}  // namespace neural_net