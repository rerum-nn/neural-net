#include "NormalRandom.h"

namespace neural_net {
NormalRandom::NormalRandom(double mean, double stddev, unsigned long seed)
    : dre_(seed), nd_(mean, stddev) {
}
double NormalRandom::Next() {
    return nd_(dre_);
}
}  // namespace neural_net