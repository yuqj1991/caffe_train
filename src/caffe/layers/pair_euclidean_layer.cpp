#include <vector>

#include "caffe/layers/euclidean_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PairEuclideanLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 2) this->output_labels_ = true;
  else                    this->output_labels_ = false;
  CHECK_EQ(bottom.size(), top.size());
  CHECK_EQ(bottom[0]->num()%2, 0);
  top[0]->Reshape(bottom[0]->num()/2, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  CHECK_EQ(bottom[0]->count()/2, top[0]->count());
  if (this->output_labels_) {
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    CHECK_EQ(bottom[1]->num(), bottom[1]->count());
    vector<int> shape(1, top[0]->num());
    top[1]->Reshape(shape);
  }
  diff_.ReshapeLike(*top[0]);
}

template <typename Dtype>
void PairEuclideanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = top[0]->count();
  const Dtype* A_data = bottom[0]->cpu_data();
  const Dtype* B_data = bottom[0]->cpu_data() + count;
  caffe_sub(
      count,
      A_data,
      B_data,
      diff_.mutable_cpu_data());
  caffe_mul(count, diff_.cpu_data(), diff_.cpu_data(), top_data);

  if (this->output_labels_) {
    const int count_label = top[1]->count();
    const Dtype* A_label = bottom[1]->cpu_data();
    const Dtype* B_label = bottom[1]->cpu_data() + count_label;
    Dtype* top_label = top[1]->mutable_cpu_data();
    for (int i = 0; i < count_label; i++) {
      top_label[i] = A_label[i] == B_label[i];
    }
  }
}

template <typename Dtype>
void PairEuclideanLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = top[0]->count();
    Dtype* diff = bottom[0]->mutable_cpu_diff();
    caffe_mul(count, top[0]->cpu_diff(), diff_.cpu_data(), diff);
    caffe_cpu_scale(count, Dtype(2), diff, diff);
    diff += count;
    caffe_mul(count, top[0]->cpu_diff(), diff_.cpu_data(), diff);
    caffe_cpu_scale(count, Dtype(-2), diff, diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(PairEuclideanLayer);
#endif

INSTANTIATE_CLASS(PairEuclideanLayer);
REGISTER_LAYER_CLASS(PairEuclidean);

}  // namespace caffe
