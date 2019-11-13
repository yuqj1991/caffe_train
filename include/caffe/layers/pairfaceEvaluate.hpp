#ifndef CAFFE_PAIRFACEVALUATE_LOSS_LAYER_HPP_
#define CAFFE_PAIRFACEVALUATE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class pairfaceEvaluateLayer : public Layer<Dtype> {
 public:
  explicit pairfaceEvaluateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top); 
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline const char* type() const { return "pairfaceEvaluate"; }

  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  }

  Blob<Dtype> diff_;  // cached for backward pass
  Blob<Dtype> diff_sq_;  // cached for backward pass
  Blob<Dtype> dist_left_;  
  Blob<Dtype> dist_right_; 

  Dtype eucildeanThresold_;
  Dtype cosThresold_;

  metricDistanceParameter_MetricType type_;
};

}  // namespace caffe

#endif  // CAFFE_CONTRASTIVE_LOSS_LAYER_HPP_
