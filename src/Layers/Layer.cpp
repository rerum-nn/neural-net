#include "Layer.h"
#include "Linear.h"
#include "ReLU.h"
#include "Sigmoid.h"
#include "Softmax.h"

namespace neural_net {

Layer Layer::DeserializeLayer(std::istream& is) {
    std::string name;
    is >> name;
    if (name == "linear") {
        Index rows, cols;
        is >> rows >> cols;

        Matrix weights(rows, cols);
        Vector bias(rows);
        for (Index i = 0; i < rows * cols; ++i) {
            double elem;
            is >> elem;
            weights(i / cols, i % cols) = elem;
        }
        for (Index i = 0; i < rows; ++i) {
            double elem;
            is >> elem;
            bias[i] = elem;
        }

        return Linear(weights, bias);
    } else if (name == "relu") {
        return ReLU();
    } else if (name == "sigmoid") {
        return Sigmoid();
    } else if (name == "softmax") {
        return Softmax();
    } else {
        assert(false && "invalid data of FNN file");
    }
}

}
