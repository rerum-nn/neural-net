#include "DataManipulate.h"

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
                data_shuffled.bottomRows(test_size), labels_shuffled.bottomRows(test_size)};
    }
    return {data.topRows(train_size), labels.topRows(train_size), data.bottomRows(test_size),
            labels.bottomRows(test_size)};
}

Matrix IntLabelsToCategorical(const Matrix& labels) {
    assert(labels.cols() == 1);
    Eigen::MatrixXi int_matrix = labels.cast<int>();
    assert((int_matrix.array() >= 0).all());
    Index max_category = int_matrix.maxCoeff() + 1;

    Matrix categorical_labels(labels.rows(), max_category);
    categorical_labels.setZero();
    for (Index i = 0; i < int_matrix.rows(); ++i) {
        categorical_labels(i, int_matrix(i, 0)) = 1;
    }
    return categorical_labels;
}

}  // namespace neural_net