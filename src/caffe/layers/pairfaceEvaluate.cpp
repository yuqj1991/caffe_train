#include <algorithm>
#include <vector>

#include "caffe/layers/pairfaceEvaluate.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void pairfaceEvaluateLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_left_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_right_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);

  eucildeanThresold_ = this->layer_param_.metric_param().eucil_threshold();
  cosThresold_ = this->layer_param_.metric_param().cos_threshold();
  type_ = this->layer_param_.metric_param().metric();
}

template <typename Dtype>
void pairfaceEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void pairfaceEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  const int channels = bottom[0]->channels();
  if(type_==metricDistanceParameter_MetricType_EUCILDISTANCE){
    caffe_sub(
        count,
        bottom[0]->cpu_data(),  // a
        bottom[1]->cpu_data(),  // b
        diff_.mutable_cpu_data());  // a_i-b_i
    caffe_sqr<Dtype>(count, diff_.cpu_data(), diff_sq_.mutable_cpu_data());
    
    Dtype squre_sum = Dtype(0.0);
    for (int i = 0; i < bottom[0]->num(); ++i) {
      squre_sum = caffe_cpu_asum<Dtype>(channels,
          diff_sq_.cpu_data() + (i*channels));
      if(squre_sum >= eucildeanThresold_)
        top[0]->mutable_cpu_data()[i] = Dtype(0.0);
      else
        top[0]->mutable_cpu_data()[i] = Dtype(1.0);
    }
  }else if(type_==metricDistanceParameter_MetricType_COSDISTANCE){
    caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_.mutable_cpu_data()); 
    caffe_sqr<Dtype>(count, bottom[0]->cpu_data(), dist_left_.mutable_cpu_data());
    caffe_sqr<Dtype>(count, bottom[1]->cpu_data(), dist_right_.mutable_cpu_data());
    Dtype top_sum = Dtype(0.0);
    Dtype bottom_left = Dtype(0.0);
    Dtype bottom_right = Dtype(0.0);
    Dtype cosvalue = Dtype(0.0);
    for (int i = 0; i < bottom[0]->num(); ++i) {
      top_sum = caffe_cpu_asum<Dtype>(channels,
          diff_.cpu_data() + (i*channels));
      bottom_left = caffe_cpu_asum<Dtype>(channels,
          dist_left_.cpu_data() + (i*channels));
      bottom_right = caffe_cpu_asum<Dtype>(channels,
          dist_right_.cpu_data() + (i*channels));
      cosvalue = Dtype(top_sum *(std::pow(bottom_left, -0.5) * std::pow(bottom_right, -0.5) + 0.00000001));
      if(cosvalue >= cosThresold_)
        top[0]->mutable_cpu_data()[i] = Dtype(1.0);
      else
        top[0]->mutable_cpu_data()[i] = Dtype(0.0);
    }
  }

}
#ifdef CPU_ONLY
STUB_GPU(pairfaceEvaluateLayer);
#endif

INSTANTIATE_CLASS(pairfaceEvaluateLayer);
REGISTER_LAYER_CLASS(pairfaceEvaluate);

}  // namespace caffe
