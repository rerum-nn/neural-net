#include "BatchSlicer.h"

#include "Random.h"

namespace neural_net {

BatchSlicer::BatchSlicer(const Matrix &data, const Matrix &labels, Index batch_size,
                         ShuffleMode mode)
    : data_(data), labels_(labels), batch_size_(batch_size), mode_(mode) {
}

BatchSlicer::BatchSlicerIterator BatchSlicer::begin() {
    return BatchSlicer::BatchSlicerIterator(&data_, &labels_, 0, batch_size_);
}

BatchSlicer::BatchSlicerIterator BatchSlicer::end() {

    return BatchSlicer::BatchSlicerIterator(&data_, &labels_, data_.rows(), batch_size_);
}

BatchSlicer::BatchSlicer(Matrix &&data, Matrix &&labels, Index batch_size, ShuffleMode mode)
    : data_(std::move(data)), labels_(std::move(labels)), batch_size_(batch_size), mode_(mode) {
}

void BatchSlicer::Reset(const Matrix &data, const Matrix &labels, Index batch_size,
                        ShuffleMode mode) {
    data_ = data;
    labels_ = labels;
    batch_size_ = batch_size;
    mode_ = mode;
}

void BatchSlicer::Reset(Matrix &&data, Matrix &&labels, Index batch_size, ShuffleMode mode) {
    data_ = std::move(data);
    labels_ = std::move(labels);
    batch_size_ = batch_size_;
    mode_ = mode;
}

void BatchSlicer::Shuffle() {
    if (mode_ == ShuffleMode::Static) {
        return;
    }
    PermutationMatrix perm = Random::Permutation(data_.rows());
    data_ = perm * data_;
    labels_ = perm * labels_;
}

BatchSlicer::BatchSlicerIterator::BatchSlicerIterator(const Matrix *data, const Matrix *labels,
                                                      Index idx, Index batch_size)
    : data_(data), labels_(labels), idx_(idx), batch_size_(batch_size) {
    assert(batch_size != 0);
    assert(data != nullptr && labels != nullptr);
    assert(data->rows() == labels->rows() && idx <= data->rows());
}

BatchSlicer::BatchSlicerIterator::Batch BatchSlicer::BatchSlicerIterator::operator*() const {
    size_t batch_size = std::min(batch_size_, data_->rows() - idx_);
    return {data_->block(idx_, 0, batch_size, data_->cols()),
            labels_->block(idx_, 0, batch_size, labels_->cols())};
}

BatchSlicer::BatchSlicerIterator &BatchSlicer::BatchSlicerIterator::operator++() {
    size_t batch_size = std::min(batch_size_, data_->rows() - idx_);
    idx_ += batch_size;
    return *this;
}

BatchSlicer::BatchSlicerIterator BatchSlicer::BatchSlicerIterator::operator++(int) {
    BatchSlicerIterator it = *this;
    size_t batch_size = std::min(batch_size_, data_->rows() - idx_);
    idx_ += batch_size;
    return it;
}

bool BatchSlicer::BatchSlicerIterator::operator!=(
    const BatchSlicer::BatchSlicerIterator &other) const {
    return idx_ != other.idx_ || data_ != other.data_ || labels_ != other.labels_ ||
           batch_size_ != other.batch_size_;
}
}  // namespace neural_net