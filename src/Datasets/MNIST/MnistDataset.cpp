#include "MnistDataset.h"

#include <fstream>
#include <iostream>

namespace neural_net {
std::tuple<Matrix, Matrix, Matrix, Matrix> MnistDataset::LoadData() {
    auto [train_data, train_labels] = ReadCsv("mnist_train.csv");
    auto [test_data, test_labels] = ReadCsv("mnist_test.csv");
    train_data /= 255;
    test_data /= 255;
    return {train_data, train_labels, test_data, test_labels};
}

std::pair<Matrix, Matrix> MnistDataset::ReadCsv(const std::string& path) {
    std::ifstream csv;
    csv.open(path);
    if (!csv) {
        std::cerr << path << " file can't be opened\n";
        return {};
    }

    std::vector<double> values;
    std::vector<double> labels;
    std::string str;
    Index rows = 0;

    if (!std::getline(csv, str)) {
        std::cerr << path << " file is empty\n";
        return {};
    }

    while (std::getline(csv, str)) {
        std::stringstream sstream(str);
        std::string value;
        std::getline(sstream, value, ',');
        labels.push_back(std::stoi(value));
        while (std::getline(sstream, value, ',')) {
            values.push_back(std::stod(value));
        }
        ++rows;
    }
    Index cols = values.size() / rows;

    return {Eigen::Map<Matrix, Eigen::Unaligned, Eigen::Stride<1, kImageSize>>(values.data(), rows,
                                                                               cols),
            Eigen::Map<Matrix>(labels.data(), rows, 1)};
}

}  // namespace neural_net