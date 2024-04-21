#pragma once

#include "../LossFunctions/LossFunction.h"
#include "../Sequential.h"
#include "../Types.h"

namespace neural_net {

class Adam {
public:
    enum class FastStart { Enable, Disable };

    Adam(double lr = 0.03, double beta_1 = 0.9, double beta_2 = 0.999,
         FastStart is_fast_start = FastStart::Enable);

    void operator()(Sequential& sequential, const Matrix& input_data, const Matrix& labels,
                    const LossFunction& loss, size_t max_epoch = 10000) const;

private:
    void UpdateParameter(const std::vector<ParametersGrad>& pack, std::vector<Matrix>& first_moment,
                         std::vector<Matrix>& second_moment, double cur_beta_1,
                         double cur_beta_2) const;

    static constexpr double kEpsilon = 1e-8;

    double learning_rate_;
    double beta_1_;
    double beta_2_;
    bool is_fast_start_;
};

}  // namespace neural_net
