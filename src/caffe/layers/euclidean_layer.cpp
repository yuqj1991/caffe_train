#include <vector>

#include "caffe/layers/euclidean_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_mul(count, diff_.cpu_data(), diff_.cpu_data(), top_data);
}

template <typename Dtype>
void EuclideanLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 2 : -2;
      caffe_mul(bottom[i]->count(), top[0]->cpu_diff(), diff_.cpu_data(), bottom[i]->mutable_cpu_diff());
      caffe_cpu_scale(bottom[i]->count(), sign, bottom[i]->cpu_diff(),
               bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLayer);
#endif

INSTANTIATE_CLASS(EuclideanLayer);
REGISTER_LAYER_CLASS(Euclidean);

}  // namespace caffe
