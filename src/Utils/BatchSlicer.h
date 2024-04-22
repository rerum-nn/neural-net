#pragma once

#include "Types.h"

namespace neural_net {

class BatchSlicer {
public:
    class BatchSlicerIterator {
    public:
        using Batch = std::pair<Matrix, Matrix>;

        BatchSlicerIterator(const Matrix* data, const Matrix* labels, Index idx = 0,
                            Index batch_size = 1);

        Batch operator*() const;
        BatchSlicerIterator& operator++();
        BatchSlicerIterator operator++(int);
        bool operator!=(const BatchSlicerIterator& other) const;

    private:
        const Matrix* data_;
        const Matrix* labels_;
        Index idx_;
        Index batch_size_;
    };

    BatchSlicer(const Matrix& data, const Matrix& labels, Index batch_size = 1,
                ShuffleMode mode = ShuffleMode::Shuffle);
    BatchSlicer(Matrix&& data, Matrix&& labels, Index batch_size = 1,
                ShuffleMode mode = ShuffleMode::Shuffle);

    void Reset(const Matrix& data, const Matrix& labels, Index batch_size = 1,
               ShuffleMode mode = ShuffleMode::Shuffle);
    void Reset(Matrix&& data, Matrix&& labels, Index batch_size = 1,
               ShuffleMode mode = ShuffleMode::Shuffle);
    void Shuffle();

    BatchSlicerIterator begin();
    BatchSlicerIterator end();

private:
    Matrix data_;
    Matrix labels_;
    Index batch_size_;
    ShuffleMode mode_;
};

}  // namespace neural_net
