#include <cmath>
#include <vector>

#include "caffe/layers/wighted_eltwise_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CudaExpSum(const Dtype* x, int size, Dtype * max_value, Dtype * sum_value){
  Dtype MaxVaule = x[0];
  Dtype sumValue = Dtype(0.f);
  // 求出每组的最大值
  CUDA_KERNEL_LOOP(index, size){
    MaxVaule = max(MaxVaule, x[index]);
  }
  *max_value = MaxVaule;
  // 每个样本组减去最大值， 计算exp，求和
  CUDA_KERNEL_LOOP(index, size){
    sumValue += exp(x[index] - MaxVaule);
  }
  *sum_value = sumValue;
}

template <typename Dtype>
__global__ void CudaNormalSum(const Dtype* x, int size,  Dtype * sum_value){
  Dtype sumValue = Dtype(0.f);
  // 每个样本组减去最大值， 计算exp，求和
  CUDA_KERNEL_LOOP(index, size){
    sumValue += x[index];
  }
  *sum_value = sumValue + 0.0001;
}

template <typename Dtype>
void WightEltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int bottom_size = bottom.size();
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
    Dtype sumValue =  Dtype(0.);
    CudaExpSum<Dtype><<<CAFFE_GET_BLOCKS(bottom_size), CAFFE_CUDA_NUM_THREADS>>>(weight, bottom_size, &maxValue, &sumValue);
    Dtype* weight_Normal_data = weight_Normal_.mutable_gpu_data();
    for (int i = 0; i < bottom_size; ++i) {
      Dtype SoftWight = std::exp(weight[i] - maxValue) / sumValue;
      weight_Normal_data[i] = SoftWight;
      caffe_axpy(count, SoftWight, bottom[i]->gpu_data(), top_data);
    }
  }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_SOFTMAX){
    Dtype sumValue = Dtype(0.);
    CudaNormalSum<Dtype><<<CAFFE_GET_BLOCKS(bottom_size), CAFFE_CUDA_NUM_THREADS>>>(weight, bottom_size, &sumValue);
    Dtype* weight_Normal_data = weight_Normal_.mutable_gpu_data();
    for (int i = 0; i < bottom_size; ++i) {
      Dtype NormalWeight = (weight[i]) / (sumValue);
      weight_Normal_data[i] = NormalWeight;
      caffe_axpy(count, NormalWeight, bottom[i]->gpu_data(), top_data);
    }
  }
}


template <typename Dtype>
void WightEltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
      int bottom_size = bottom.size();
      const Dtype *weight = this->blobs_[0]->gpu_data();
      const Dtype *top_diff = top[0]->gpu_diff();
      const Dtype *top_data = top[0]->gpu_data();
      for (int j = 0; j < bottom_size; ++j) {
        if (propagate_down[j]) {
          Dtype* bottom_diff = bottom[j]->mutable_gpu_diff();
          const int count = bottom[j]->count();
          if(fusionOp_ == WightedEltwiseParameter_FusionOp_NORMAL){
            for (int i = 0; i < count; ++i) {
              bottom_diff[i] = top_diff[i] * weight[j];
            }
          }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_SOFTMAX){
            const Dtype* weight_Normal_data = weight_Normal_.gpu_data();
            for (int i = 0; i < count; ++i) {
              bottom_diff[i] = top_diff[i] * weight_Normal_data[j];
            }
          }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_FASTER){
            const Dtype* weight_Normal_data = weight_Normal_.gpu_data();
            for (int i = 0; i < count; ++i) {
              bottom_diff[i] = top_diff[i] * weight_Normal_data[j];
            }
          }
        }
      }
      
      if (this->param_propagate_down_[0]) {
        Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
        Dtype* temp_data = temp_diff_.mutable_gpu_data();
        if(fusionOp_ == WightedEltwiseParameter_FusionOp_NORMAL){
          for (int j = 0; j < bottom_size; ++j) {
            const int count = bottom[j]->count();
            const Dtype* bottom_data = bottom[j]->gpu_data();
            caffe_mul(count, bottom_data, top_diff, temp_data);
            caffe_gpu_asum(count, temp_diff_.gpu_data(), &(weight_diff[j]));
          }
        }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_SOFTMAX){
          const Dtype* weight_Normal_data = weight_Normal_.gpu_data();
          for (int j = 0; j < bottom_size; ++j) {
            const int count = bottom[j]->count();
            const Dtype* bottom_data = bottom[j]->gpu_data();
            caffe_sub(count, bottom_data, top_data, temp_data);
            caffe_mul(count, temp_data, top_diff, temp_data);
            caffe_gpu_asum(count, temp_diff_.gpu_data(), &(weight_diff[j]));
            weight_diff[j] *= weight_Normal_data[j] ;
          }
        }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_FASTER){
          Dtype sumValue = Dtype(0.);
          CudaNormalSum<Dtype><<<CAFFE_GET_BLOCKS(bottom_size), CAFFE_CUDA_NUM_THREADS>>>(weight, bottom_size, &sumValue);
          for (int j = 0; j < bottom_size; ++j) {
            const int count = bottom[j]->count();
            const Dtype* bottom_data = bottom[j]->gpu_data();
            caffe_sub(count, bottom_data, top_data, temp_data);
            caffe_mul(count, temp_data, top_diff, temp_data);
            caffe_gpu_asum(count, temp_diff_.gpu_data(), &(weight_diff[j]));
            weight_diff[j] /= sumValue ;
          }
        }
      }
}

INSTANTIATE_LAYER_GPU_FUNCS(WightEltwiseLayer);


}  // namespace caffe
