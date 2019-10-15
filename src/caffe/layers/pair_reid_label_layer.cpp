#include <vector>

#include "caffe/layers/euclidean_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PairReidLabelLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num()%2, 0);
  CHECK_EQ(bottom[0]->num(), bottom[0]->count());
  vector<int> shape(1, bottom[0]->num()/2);
  top[0]->Reshape(shape);
}

template <typename Dtype>
void PairReidLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count_label = top[0]->count();
  const Dtype* A_label = bottom[0]->cpu_data();
  const Dtype* B_label = bottom[0]->cpu_data() + count_label;
  Dtype* top_label = top[0]->mutable_cpu_data();
  for (int i = 0; i < count_label; i++) {
    top_label[i] = A_label[i] == B_label[i];
  }
}

template <typename Dtype>
void PairReidLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << "PairReidLabelLayer Can not BP";
  }
}

#ifdef CPU_ONLY
STUB_GPU(PairReidLabelLayer);
#endif

INSTANTIATE_CLASS(PairReidLabelLayer);
REGISTER_LAYER_CLASS(PairReidLabel);

}  // namespace caffe
