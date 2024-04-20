#include "Adam.h"

namespace neural_net {
Adam::Adam(double lr, double beta_1, double beta_2, FastStart is_fast_start)
    : learning_rate_(lr),
      beta_1_(beta_1),
      beta_2_(beta_2),
      is_fast_start_(is_fast_start == FastStart::Enable) {
}

void Adam::operator()(Network& network, const Matrix& input_data, const Matrix& labels,
                      const LossFunction& loss, size_t max_epoch) const {
    std::vector<Layer>& layers = network.GetLayers();

    std::vector<std::vector<Matrix>> first_moment(layers.size());
    std::vector<std::vector<Matrix>> second_moment(layers.size());
    double cur_beta_1 = beta_1_;
    double cur_beta_2 = beta_2_;
    if (is_fast_start_) {
        cur_beta_1 = 0;
        cur_beta_2 = 0;
    }

    for (size_t epoch = 1; epoch <= max_epoch; ++epoch) {
        for (Index batch = 0; batch < input_data.cols(); ++batch) {
            Vector label = labels.col(batch);
            Vector output = network.Predict(input_data.col(batch));

            RowVector nabla = loss->LossGradient(output, label);
            for (size_t i = 0; i < layers.size(); ++i) {
                size_t pos = layers.size() - 1 - i;
                Layer& layer = layers[pos];
                UpdateParameter(layer->GetGradients(nabla), first_moment[pos], second_moment[pos],
                                cur_beta_1, cur_beta_2);
                nabla = layer->BackPropagation(nabla);
            }
            cur_beta_1 *= beta_1_;
            cur_beta_2 *= beta_2_;
        }
    }
}

void Adam::UpdateParameter(const std::vector<ParametersGrad>& pack,
                           std::vector<Matrix>& first_moment, std::vector<Matrix>& second_moment,
                           double cur_beta_1, double cur_beta_2) const {
    if (first_moment.empty() && second_moment.empty()) {
        for (size_t i = 0; i < pack.size(); ++i) {
            const ParametersGrad& param = pack[i];
            Matrix m;
            Matrix v;
            m = (1 - beta_1_) * param.grad;
            v = (1 - beta_2_) * param.grad.cwiseProduct(param.grad);
            first_moment.push_back(m);
            second_moment.push_back(v);
            Matrix m_hat = m / (1 - cur_beta_1);
            Matrix v_hat = v / (1 - cur_beta_2);
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
        Matrix m_hat = m / (1 - cur_beta_1);
        Matrix v_hat = v / (1 - cur_beta_2);
        param.param -=
            learning_rate_ * (m_hat.array() / ((v_hat.array() + kEpsilon).sqrt())).matrix();
    }
}
}  // namespace neural_net