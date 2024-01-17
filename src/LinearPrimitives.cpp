#include "LinearPrimitives.h"

#include <algorithm>
#include <cassert>

namespace neural_net {

DataVector& operator+=(DataVector& vector, const DataVector& other) {
    assert(vector.size() == other.size() && "two Vectors must have consistent sizes");
    std::transform(vector.begin(), vector.end(), other.begin(), vector.begin(),
                   [](double d1, double d2) { return d1 + d2; });
    return vector;
}
DataVector operator+(const DataVector& vector, const DataVector& other) {
    DataVector res(vector);
    res += other;
    return res;
}

DataVector operator*(const WeightMatrix& weight_matrix, const DataVector& data_vector) {
    assert(!weight_matrix.empty() && "WeightMatrix can't be empty");
    assert(weight_matrix.at(0).size() == data_vector.size() &&
           "Matrix and Vector must have consistent sizes.");

    DataVector res(data_vector.size());

    std::transform(weight_matrix.begin(), weight_matrix.end(), res.begin(),
                   [&data_vector](const std::vector<double>& line) {
                       double res = 0;
                       for (size_t i = 0; i < line.size(); ++i) {
                           res += line[i] * data_vector[i];
                       }
                       return res;
                   });

    return res;
}
DataVector operator*(const DataVector& data_vector, const WeightMatrix& weight_matrix) {
    assert(!weight_matrix.empty() && "WeightMatrix can't be empty");
    assert(weight_matrix.size() == data_vector.size() &&
           "Matrix and Vector must have consistent sizes.");

    DataVector res(weight_matrix.at(0).size(), 0);

    for (size_t i = 0; i < data_vector.size(); ++i) {
        for (size_t j = 0; j < weight_matrix.at(0).size(); ++j) {
            res[j] += data_vector[i] * weight_matrix.at(i).at(j);
        }
    }

    return res;
}
WeightMatrix& operator+=(WeightMatrix& first, const WeightMatrix& second) {
    assert(!(first.empty() || second.empty()) && "Weight Matrices can't be empty");
    assert(first.size() == second.size() && first.at(0).size() == second.at(0).size() &&
           "Two Matrices must have consistent sizes");

    for (size_t i = 0; i < first.size(); ++i) {
        for (size_t j = 0; j < first.at(0).size(); ++j) {
            first.at(i).at(j) += second.at(i).at(j);
        }
    }

    return first;
}
WeightMatrix operator+(const WeightMatrix& first, const WeightMatrix& second) {
    WeightMatrix res(first);
    res += second;
    return res;
}

}  // namespace neural_net
