#include "../Types.h"

namespace neural_net {

class Softmax {
public:
    Vector Apply(const Vector& input_vector);
    std::vector<ParametersGrad> GetGradients(const RowVector& loss);
    RowVector BackPropagation(const RowVector& loss) const;

private:
    Vector exp_vector_;
};

}  // namespace neural_net
