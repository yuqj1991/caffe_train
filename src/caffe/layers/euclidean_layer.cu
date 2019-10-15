#include <vector>

#include "caffe/layers/euclidean_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  caffe_gpu_mul(count, diff_.gpu_data(), diff_.gpu_data(), top_data);
}

template <typename Dtype>
void EuclideanLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 2 : -2;
      caffe_gpu_mul(bottom[i]->count(), top[0]->gpu_diff(), diff_.gpu_data(), bottom[i]->mutable_gpu_diff());
      caffe_gpu_scale(bottom[i]->count(), sign, bottom[i]->gpu_diff(),
              bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLayer);

}  // namespace caffe
