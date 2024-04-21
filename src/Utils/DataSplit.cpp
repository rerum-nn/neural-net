#include "DataSplit.h"

#include "Random.h"

namespace neural_net {

std::tuple<Matrix, Matrix, Matrix, Matrix> TrainTestSplit(const Matrix& data, const Matrix& labels,
                                                          double train_ratio, ShuffleMode mode) {
    assert(data.rows() == labels.rows());
    size_t train_size = static_cast<int>(data.rows() * train_ratio);
    size_t test_size = data.rows() - train_size;
    if (mode == ShuffleMode::Shuffle) {
        PermutationMatrix perm = Random::Permutation(data.rows());
        Matrix data_shuffled = perm * data;
        Matrix labels_shuffled = perm * labels;
        return {data_shuffled.topRows(train_size), labels_shuffled.topRows(train_size),
                data_shuffled.bottomRows(train_size), labels_shuffled.bottomRows(test_size)};
    }
    return {data.topRows(train_size), labels.topRows(train_size), data.bottomRows(train_size),
            labels.bottomRows(test_size)};
}

}  // namespace neural_net