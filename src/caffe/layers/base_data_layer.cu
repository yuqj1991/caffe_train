#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

template <typename Dtype>
void ReidPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // CHECK
  CHECK_EQ(top[0]->count(), prefetch_current_->data_.count()*2);
  // Reshape to loaded data.
  top[0]->Reshape(prefetch_current_->data_.num()*2, prefetch_current_->data_.channels(), prefetch_current_->data_.height(), prefetch_current_->data_.width());
  // Copy the data
  caffe_copy(prefetch_current_->data_.count(),  prefetch_current_->data_.gpu_data(),  top[0]->mutable_gpu_data());
  caffe_copy(prefetch_current_->datap_.count(), prefetch_current_->datap_.gpu_data(), top[0]->mutable_gpu_data()+prefetch_current_->data_.count());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    vector<int> shape = prefetch_current_->label_.shape();
    CHECK_LT(shape.size(), 2);
    CHECK_EQ(top[1]->count(), prefetch_current_->label_.count()*2);
    shape[0] *= 2;
    top[1]->Reshape(shape);
    // Copy the labels.
    caffe_copy(prefetch_current_->label_.count(),  prefetch_current_->label_.gpu_data(),  top[1]->mutable_gpu_data());
    caffe_copy(prefetch_current_->labelp_.count(), prefetch_current_->labelp_.gpu_data(), top[1]->mutable_gpu_data()+prefetch_current_->label_.count());
  }
}

template <typename Dtype>
void ImageDataPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  pairBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(), top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    top[2]->ReshapeLike(batch->labelSample_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(), top[1]->mutable_gpu_data());
    caffe_copy(batch->labelSample_.count(), batch->labelSample_.gpu_data(), top[2]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(ReidPrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(ImageDataPrefetchingDataLayer);
}  // namespace caffe
