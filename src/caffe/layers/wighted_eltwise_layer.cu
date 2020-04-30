#include <cmath>
#include <vector>

#include "caffe/layers/wighted_eltwise_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ Dtype CudaExpSum(const Dtype* x, int size, Dtype * max_value){
  Dtype MaxVaule = x[0];
  Dtype sumValue = Dtype(0.f);
  // 求出每组的最大值
  CUDA_KERNEL_LOOP(index, size){
    MaxVaule = std::max(MaxVaule, x[index]);
  }
  *max_value = MaxVaule;
  // 每个样本组减去最大值， 计算exp，求和
  CUDA_KERNEL_LOOP(index, size){
    sumValue += std::exp(x[index] - MaxVaule);
  }
  return sumValue;
}

template <typename Dtype>
__global__ Dtype CudaNormalSum(const Dtype* x, int size){
  Dtype sumValue = Dtype(0.f);
  // 每个样本组减去最大值， 计算exp，求和
  CUDA_KERNEL_LOOP(index, size){
    sumValue += x[index];
  }
  return sumValue + 0.0001;
}

template <typename Dtype>
void WightEltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int bottome_size = bottom.size();
  const int count = top[0]->count();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_set(count, Dtype(0), top_data);
  if(fusionOp_ == WightedEltwiseParameter_FusionOp_NORMAL){
    for (int i = 0; i < bottom_size; ++i) {
      caffe_axpy(count, weight[i], bottom[i]->gpu_data(), top_data);
    }
  }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_SOFTMAX){
    Dtype maxValue = Dtype(0.);
    Dtype sumValue = CudaExpSum<Dtype><<<CAFFE_GET_BLOCKS(bottome_size), CAFFE_CUDA_NUM_THREADS>>>(weight, bottome_size, &maxValue);
    Dtype* weight_Normal_data = weight_Normal_.mutable_gpu_data();
    for (int i = 0; i < bottom_size; ++i) {
      Dtype SoftWight = std::exp(weight[i] - maxValue) / sumValue;
      weight_Normal_data[i] = SoftWight;
      caffe_axpy(count, SoftWight, bottom[i]->gpu_data(), top_data);
    }
  }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_SOFTMAX){
    Dtype sumValue = CudaNormalSum<Dtype><<<CAFFE_GET_BLOCKS(bottome_size), CAFFE_CUDA_NUM_THREADS>>>(weight, bottome_size);
    Dtype* weight_Normal_data = weight_Normal_.mutable_gpu_data();
    for (int i = 0; i < bottom_size; ++i) {
      Dtype NormalWeight = (weight[i]) / (sumValue);
      weight_Normal_data[i] = NormalWeight;
      caffe_axpy(count, NormalWeight, bottom[i]->gpu_data(), top_data);
    }
  }
}

template <typename Dtype>
__global__ void SwishBackward(const int n, const Dtype* in_diff,
    const Dtype* bottom_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = (1. / (1. + exp(-bottom_data[index])) ) * bottom_data[index];
    out_diff[index] = in_diff[index] * (sigmoid_x * (1 - sigmoid_x) * bottom_data[index] + sigmoid_x);
  }
}

template <typename Dtype>
void WightEltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    SwishBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WightEltwiseLayer);


}  // namespace caffe
