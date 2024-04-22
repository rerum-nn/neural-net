#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;

int main() {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(3, 3);

    Index i, j;
    double s = matrix.maxCoeff(&i, &j);

    std::cout << s << std::endl;
    std::cout << matrix(i, j) << std::endl;
    std::cout << i << ' ' << j << std::endl;

    for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
            std::cout << matrix(k, l) << ' ';
        }
        std::cout << std::endl;
    }
}
