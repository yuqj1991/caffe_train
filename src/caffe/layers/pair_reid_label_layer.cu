#include <vector>

#include "caffe/layers/euclidean_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReidLabelDiffForward(const int n, const Dtype* A_in, const Dtype* B_in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = Dtype(A_in[index] == B_in[index]);
  }
}

template <typename Dtype>
void PairReidLabelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count_label = top[0]->count();
  const Dtype* A_label = bottom[0]->gpu_data();
  const Dtype* B_label = bottom[0]->gpu_data() + count_label;
  Dtype* top_label = top[0]->mutable_gpu_data();
  ReidLabelDiffForward<Dtype><<<CAFFE_GET_BLOCKS(count_label), CAFFE_CUDA_NUM_THREADS>>>(
      count_label, A_label, B_label, top_label);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void PairReidLabelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << "PairReidLabelLayer Can not BP";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PairReidLabelLayer);

}  // namespace caffe
