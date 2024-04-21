#pragma once

#include <tuple>

#include "Types.h"

namespace neural_net {

class MnistDataset {
public:
    std::tuple<Matrix, Matrix, Matrix, Matrix> LoadData();

private:
    static constexpr size_t kImageSize = 784;

    static std::pair<Matrix, Matrix> ReadCsv(const std::string& path);
};

}  // namespace neural_net
