#include "Network.h"

#include <cassert>
#include <ranges>
#include <string>

namespace neural_net {

Network::Network(std::initializer_list<Layer> layers) : layers_(layers) {
    assert(layers.size() >= 1 &&
           "there should be at least one layer in a network");
}

Vector Network::Predict(const Vector& input_vector) {
    Vector iteration = input_vector;

    for (Layer& layer : layers_) {
        iteration = layer->Apply(iteration);
    }

    return iteration;
}

void Network::Fit(const Matrix& input_data, const Matrix& expected_answers,
                  const FitParameters& fit_parameters) {
    for (size_t epoch = 1; epoch <= fit_parameters.max_epoch; ++epoch) {
        for (Index batch = 0; batch < input_data.rows(); ++batch) {
            Vector iteration = Predict(input_data.row(batch));

            RowVector nabla =
                fit_parameters.learning_rate *
                fit_parameters.loss_function->LossGradient(iteration, expected_answers);
            for (Layer& layer : std::ranges::reverse_view(layers_)) {
                nabla = layer->Fit(nabla);
            }
        }
    }
}

}  // namespace neural_net
