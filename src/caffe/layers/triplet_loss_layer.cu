#include "caffe/layers/triplet_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::ComputeDiff_gpu(const Dtype *x_1,
    const Dtype *x_2, const Dtype x_1_norm, const Dtype x_2_norm,
    const Dtype inner_val, Dtype *x_1_diff) {
    caffe_gpu_scale(feature_dim_, Dtype(1) / (x_1_norm * x_2_norm),
        x_2, x_1_diff);
    Dtype x_1_norm_cubic = x_1_norm * x_1_norm * x_1_norm;
    caffe_gpu_axpby(feature_dim_, -inner_val / (x_1_norm_cubic * x_2_norm),
        x_1, Dtype(1), x_1_diff);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  this->Forward_cpu(bottom, top);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  this->Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);

}
