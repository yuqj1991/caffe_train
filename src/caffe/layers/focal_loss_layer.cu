#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void focalSoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts, float gamma, float alpha) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      Dtype prob_a = prob_data[n * dim + label_value * spatial_dim + s];
      Dtype b = powf(1- prob_a, gamma);
      loss[index] = -log(max(prob_a,
                      Dtype(FLT_MIN))) * b * alpha;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void focalSoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is not used for anything until it is overwritten
    // on the backward pass, we use it here to avoid having to allocate new GPU
    // memory to accumulate intermediate results in the kernel.
    Dtype* loss_data = bottom[0]->mutable_gpu_diff();
    // Similarly, this memory is never used elsewhere, and thus we can use it
    // to avoid having to allocate additional GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    focalSoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts, gamma_, alpha_);
    Dtype loss;
    caffe_gpu_asum(nthreads, loss_data, &loss);
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, outer_num_, inner_num_, valid_count);
    top[0]->mutable_cpu_data()[0] = loss / normalizer;
    if (top.size() == 2) {
      top[1]->ShareData(prob_);
    }
}

template <typename Dtype>
__global__ void focalSoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, const Dtype* prob_data, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts, float gamma, float alpha) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    Dtype focaldiff = 0;
    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      Dtype prob_a = prob_data[n * dim + label_value * spatial_dim + s];
      for(int c = 0; c < channels; ++c){
        if(c == label_value){
          Dtype diff_element = std::pow((1 - prob_a), gamma);
          Dtype diff_element_mutal = gamma *
                                    prob_a*log(max(prob_a,Dtype(FLT_MIN))) + prob_a -1;
          focaldiff = diff_element * diff_element_mutal * alpha;
        }else{
          Dtype pc = prob_data[n * dim + c * spatial_dim + s];
          Dtype diff_element = std::pow((1 - prob_a), gamma -1)*pc;
          Dtype diff_element_mutal =  1 - prob_a - gamma *
                                   prob_a*log(max(prob_a,Dtype(FLT_MIN)));
          focaldiff = diff_element * diff_element_mutal * alpha;
        }
        bottom_diff[n * dim + c * spatial_dim + s] = focaldiff;
      }
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void focalSoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
                << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    Dtype* counts = prob_.mutable_gpu_diff();
    focalSoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, prob_data, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts, gamma_, alpha_);

    Dtype valid_count = -1;
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, outer_num_, inner_num_, valid_count);
    const Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(focalSoftmaxWithLossLayer);

}  // namespace caffe
