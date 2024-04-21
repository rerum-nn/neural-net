#pragma once

#include "Types.h"

namespace neural_net {

class BatchSlicer {
public:
    BatchSlicer(const Matrix* data, const Matrix* labels, size_t batch_size = 1);

    class BatchSlicerIterator {
    public:
        using Batch = std::pair<Matrix, Matrix>;

        BatchSlicerIterator(const Matrix* data, const Matrix* labels, size_t idx = 0,
                            size_t batch_size = 1);

        Batch operator*() const;
        BatchSlicerIterator& operator++();
        BatchSlicerIterator operator++(int);
        bool operator!=(const BatchSlicerIterator& other) const;

    private:
        const Matrix* data_;
        const Matrix* labels_;
        size_t idx_;
        size_t batch_size_;
    };

    BatchSlicerIterator begin();
    BatchSlicerIterator end();

private:
    const Matrix* data_;
    const Matrix* labels_;
    size_t batch_size_;
};

}  // namespace neural_net
