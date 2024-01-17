#include "Layer.h"

#include <cassert>

namespace neural_net {

Layer::Layer(size_t input, size_t output, ActivationFunction func, RandomGenerator&& generator)
    : weights_(input, std::vector<double>(output, 0)),
      bias_(output, 0),
      activation_func_(std::move(func)) {
    std::for_each(weights_.begin(), weights_.end(), [&generator](std::vector<double>& line) {
        std::for_each(line.begin(), line.end(),
                      [&generator](double& d) { d = generator->Next(); });
    });
    std::for_each(bias_.begin(), bias_.end(), [&generator](double& d) { d = generator->Next(); });
}
DataVector Layer::Forward(const DataVector& input_vector) const {
    DataVector res = input_vector * weights_ + bias_;
    activation_func_->Apply(&res);
    return res;
}
void Layer::Forward(DataVector* input_vector) const {
    assert(input_vector && "input_vector in Layer Forward method can`t be nullptr");
    *input_vector = *input_vector * weights_ + bias_;
    activation_func_->Apply(input_vector);
}

}  // namespace neural_net