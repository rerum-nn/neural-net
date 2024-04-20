#include "Optimizers.h"

#include "Adam.h"
#include "SGD.h"

namespace neural_net {

Optimizer Optimizers::SGD(double lr) {
    return neural_net::SGD(lr);
}

Optimizer Optimizers::Momentum(double lr, double momentum) {
    return neural_net::SGD(lr, momentum);
}

Optimizer Optimizers::AdaGrad(double lr) {
    return neural_net::Adam(lr, 0, 0, false);
}

Optimizer Optimizers::RMSProp(double lr, double rho) {
    return neural_net::Adam(lr, 0, rho, false);
}

Optimizer Optimizers::Adam(double lr, double beta_1, double beta_2) {
    return neural_net::Adam(lr, beta_1, beta_2);
}

}