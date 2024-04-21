#include "BatchSlicer.h"

namespace neural_net {

BatchSlicer::BatchSlicer(const Matrix *data, const Matrix *labels, size_t batch_size)
    : data_(data), labels_(labels), batch_size_(batch_size) {
    assert(data != nullptr && labels != nullptr);
}

BatchSlicer::BatchSlicerIterator BatchSlicer::begin() {
    return BatchSlicer::BatchSlicerIterator(data_, labels_, 0, batch_size_);
}

BatchSlicer::BatchSlicerIterator BatchSlicer::end() {

    return BatchSlicer::BatchSlicerIterator(data_, labels_, data_->rows(), batch_size_);
}

BatchSlicer::BatchSlicerIterator::BatchSlicerIterator(const Matrix *data, const Matrix *labels,
                                                      size_t idx, size_t batch_size)
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