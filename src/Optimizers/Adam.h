#pragma once

#include "Layers/Linear.h"
#include "LossFunctions/LossFunction.h"
#include "Types.h"

namespace neural_net {

class Adam {
public:
    enum class FastStart { Enable, Disable };

    Adam(double lr = 0.03, double beta_1 = 0.9, double beta_2 = 0.999,
         FastStart is_fast_start = FastStart::Enable);

    void InitParameters(const std::vector<Linear>& layers);
    void Update(const UpdatePack& pack, size_t layer_id);
    void BatchCallback();
    void EpochCallback(size_t epoch, size_t max_epoch);

private:
    static constexpr double kEpsilon = 1e-8;

    double learning_rate_;
    double beta_1_;
    double beta_2_;
    bool is_fast_start_;

    std::vector<GradsPack> momentums_;
    std::vector<GradsPack> velocities_;
    double cur_beta_1_;
    double cur_beta_2_;
};

}  // namespace neural_net
