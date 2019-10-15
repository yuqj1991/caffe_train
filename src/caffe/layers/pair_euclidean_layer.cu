#include <vector>

#include "caffe/layers/euclidean_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LabelDiffForward(const int n, const Dtype* A_in, const Dtype* B_in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = Dtype(A_in[index] == B_in[index]);
  }
}

template <typename Dtype>
void PairEuclideanLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  const Dtype* A_data = bottom[0]->gpu_data();
  const Dtype* B_data = bottom[0]->gpu_data() + count;
  caffe_gpu_sub(
      count,
      A_data,
      B_data,
      diff_.mutable_gpu_data());
  caffe_gpu_mul(count, diff_.gpu_data(), diff_.gpu_data(), top_data);

  DLOG(INFO) << "PairEuclideanLayer : Forward_gpu : bottom : " << bottom[0]->shape_string() << ", top : " << top[0]->shape_string();
  if (this->output_labels_) {
    const int count_label = top[1]->count();
    const Dtype* A_label = bottom[1]->gpu_data();
    const Dtype* B_label = bottom[1]->gpu_data() + count_label;
    Dtype* top_label = top[1]->mutable_gpu_data();
    LabelDiffForward<Dtype><<<CAFFE_GET_BLOCKS(count_label), CAFFE_CUDA_NUM_THREADS>>>(
      count_label, A_label, B_label, top_label);
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
void PairEuclideanLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = top[0]->count();
    Dtype* diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_mul(count, top[0]->gpu_diff(), diff_.gpu_data(), diff);
    caffe_gpu_scale(count, Dtype(2), diff, diff);
    diff += count;
    caffe_gpu_mul(count, top[0]->gpu_diff(), diff_.gpu_data(), diff);
    caffe_gpu_scale(count, Dtype(-2), diff, diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PairEuclideanLayer);

}  // namespace caffe
