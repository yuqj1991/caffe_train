#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts, float gamma_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      Dtype prob_a = prob_data[n * dim + label_value * spatial_dim + s];
      Dtype b = 0.f;
      b = powf(1- prob_a, gamma_);
      loss[index] = -log(max(prob_a,
                      Dtype(FLT_MIN))) * b;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void focalSoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  this->Forward_cpu(bottom, top);
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, const Dtype* prob_data, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts, float gamma_) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      Dtype prob_a = prob_data[n * dim + label_value * spatial_dim + s];
      Dtype diff_element = 0.f;
      diff_element = powf(1 - prob_a, gamma_);
      Dtype diff_element_mutal =  1 - prob_a - gamma_ *
                                    prob_a*log(max( prob_a,Dtype(FLT_MIN)));
      bottom_diff[n * dim + label_value * spatial_dim + s] = diff_element * diff_element_mutal;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void focalSoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  this->Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(focalSoftmaxWithLossLayer);

}  // namespace caffe
