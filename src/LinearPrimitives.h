#pragma once

#include <vector>

namespace neural_net {

// TODO: fully shit, will be replaced to Eigen

using DataVector = std::vector<double>;
using WeightMatrix = std::vector<std::vector<double>>;

DataVector& operator+=(DataVector& vector, const DataVector& other);
DataVector operator+(const DataVector& vector, const DataVector& other);

DataVector operator*(const WeightMatrix& weight_matrix, const DataVector& data_vector);
DataVector operator*(const DataVector& data_vector, const WeightMatrix& weight_matrix);

WeightMatrix operator+(const WeightMatrix& first, const WeightMatrix& second);
WeightMatrix& operator+=(WeightMatrix& first, const WeightMatrix& second);

} // namespace neural_net
