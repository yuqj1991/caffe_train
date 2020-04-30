#include <cmath>
#include <vector>

#include "caffe/layers/wighted_eltwise_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype expSum(const Dtype* x, int size, Dtype * max_value){
  Dtype MaxVaule = x[0];
  Dtype sumValue = Dtype(0.f);
  // 求出每组的最大值
  for(int c = 0; c< size; c++){
    MaxVaule = std::max(MaxVaule, x[c]);
  }
  *max_value = MaxVaule;
  // 每个样本组减去最大值， 计算exp，求和
  for(int c = 0; c< size; c++){
    sumValue += std::exp(x[c] - MaxVaule);
  }
  return sumValue;
}

template <typename Dtype>
inline Dtype NormalSum(const Dtype* x, int size){
  Dtype sumValue = Dtype(0.f);
  // 每个样本组减去最大值， 计算exp，求和
  for(int c = 0; c< size; c++){
    sumValue += x[c];
  }
  return sumValue + 0.0001;
}

void WightEltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
  this->blobs_.resize(1);
    // Initialize the weights
    vector<int> weight_shape(1, bottom.size());
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.wighted_eltwise_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  fusionOp_ = this->layer_param_.wighted_eltwise_param().operation();
}


template <typename Dtype>
void WightEltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  for(unsigned i = 0; i < bottom.size(); i++){
    CHECK_EQ(top_shape, bottom[i]->shape());
  }
  CHECK_GE(bottom.size(), 2);
  top[0]->Reshape(top_shape);
  temp_diff_.Reshape(top_shape);
  weight_Normal_.Reshape(this->blobs[0]);
}


template <typename Dtype>
void WightEltwiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int bottome_size = bottom.size();
  const int count = top[0]->count();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(count, Dtype(0), top_data);
  if(fusionOp_ == WightedEltwiseParameter_FusionOp_NORMAL){
    for (int i = 0; i < bottom_size; ++i) {
      caffe_axpy(count, weight[i], bottom[i]->cpu_data(), top_data);
    }
  }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_SOFTMAX){
    Dtype maxValue = Dtype(0.);
    Dtype sumValue = expSum<Dtype>(weight, bottome_size, &maxValue);
    Dtype* weight_Normal_data = weight_Normal_.mutable_cpu_data();
    for (int i = 0; i < bottom_size; ++i) {
      Dtype SoftWight = std::exp(weight[i] - maxValue) / sumValue;
      weight_Normal_data[i] = SoftWight;
      caffe_axpy(count, SoftWight, bottom[i]->cpu_data(), top_data);
    }
  }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_FASTER){
    Dtype sumValue = NormalSum<Dtype>(weight, bottome_size);
    Dtype* weight_Normal_data = weight_Normal_.mutable_cpu_data();
    for (int i = 0; i < bottom_size; ++i) {
      Dtype NormalWeight = (weight[i]) / (sumValue);
      weight_Normal_data[i] = NormalWeight;
      caffe_axpy(count, NormalWeight, bottom[i]->cpu_data(), top_data);
    }
  }
}

template <typename Dtype>
void WightEltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int bottome_size = bottom.size();
  const Dtype *weight = this->blobs_[0]->cpu_data();
  const Dtype *top_diff = top[0]->cpu_diff();
  const Dtype *top_data = top[0]->cpu_data();
  for (int j = 0; j < bottom_size; ++j) {
    if (propagate_down[j]) {
      Dtype* bottom_diff = bottom[j]->mutable_cpu_diff();
      const int count = bottom[j]->count();
      if(fusionOp_ == WightedEltwiseParameter_FusionOp_NORMAL){
        for (int i = 0; i < count; ++i) {
          bottom_diff[i] = top_diff[i] * weight[j];
        }
      }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_SOFTMAX){
        const Dtype* weight_Normal_data = weight_Normal_.cpu_data();
        for (int i = 0; i < count; ++i) {
          bottom_diff[i] = top_diff[i] * weight_Normal_data[j];
        }
      }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_FASTER){
        const Dtype* weight_Normal_data = weight_Normal_.cpu_data();
        for (int i = 0; i < count; ++i) {
          bottom_diff[i] = top_diff[i] * weight_Normal_data[j];
        }
      }
    }
  }
  
  if (this->param_propagate_down_[0]) {
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* temp_data = temp_diff_.mutable_cpu_data();
    if(fusionOp_ == WightedEltwiseParameter_FusionOp_NORMAL){
      for (int j = 0; j < bottom_size; ++j) {
        const int count = bottom[j]->count();
        const Dtype* bottom_data = bottom[j]->cpu_data();
        caffe_mul(count, bottom_data, top_diff, temp_data);
        weight_diff[j] = caffe_cpu_asum(count, temp_diff_.cpu_data());
      }
    }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_SOFTMAX){
      const Dtype* weight_Normal_data = weight_Normal_.cpu_data();
      for (int j = 0; j < bottom_size; ++j) {
        const int count = bottom[j]->count();
        const Dtype* bottom_data = bottom[j]->cpu_data();
        caffe_mul(count, bottom_data - top_data, top_diff, temp_data);
        weight_diff[j] = caffe_cpu_asum(count, temp_diff_.cpu_data());
        weight_diff[j] *= weight_Normal_data[j] ;
      }
    }else if(fusionOp_ == WightedEltwiseParameter_FusionOp_FASTER){
      const Dtype* weight_Normal_data = weight_Normal_.cpu_data();
      Dtype sumValue = NormalSum<Dtype>(weight, bottome_size);
      for (int j = 0; j < bottom_size; ++j) {
        const int count = bottom[j]->count();
        const Dtype* bottom_data = bottom[j]->cpu_data();
        caffe_mul(count, bottom_data - top_data, top_diff, temp_data);
        weight_diff[j] = caffe_cpu_asum(count, temp_diff_.cpu_data());
        weight_diff[j] /= sumValue ;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WightEltwiseLayer);
#endif

INSTANTIATE_CLASS(WightEltwiseLayer);


}  // namespace caffe
