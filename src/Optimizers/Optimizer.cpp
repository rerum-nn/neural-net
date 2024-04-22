#include "Optimizer.h"

#include "Adam.h"
#include "SGD.h"

namespace neural_net {

Optimizer Optimizer::SGD(double lr) {
    return neural_net::SGD(lr);
}

Optimizer Optimizer::Momentum(double lr, double momentum) {
    return neural_net::SGD(lr, momentum);
}

Optimizer Optimizer::AdaGrad(double lr) {
    return neural_net::Adam(lr, 0, 0, neural_net::Adam::FastStart::Disable);
}

Optimizer Optimizer::RMSProp(double lr, double rho) {
    return neural_net::Adam(lr, 0, rho, neural_net::Adam::FastStart::Disable);
}

Optimizer Optimizer::Adam(double lr, double beta_1, double beta_2) {
    return neural_net::Adam(lr, beta_1, beta_2);
}

}  // namespace neural_net
