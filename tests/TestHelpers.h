#pragma once

#include "Types.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

using neural_net::Index;
using neural_net::Matrix;
using testing::FloatNear;

inline void CheckCloseMatrix(const Matrix& lhs, const Matrix& rhs, float abs_error = 1e-7) {
    ASSERT_TRUE(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    for (Index i = 0; i < lhs.rows(); ++i) {
        for (Index j = 0; j < lhs.cols(); ++j) {
            ASSERT_THAT(lhs(i, j), FloatNear(rhs(i, j), abs_error));
        }
    }
}
