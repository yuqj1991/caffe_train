#ifndef CAFFE_SIGMOID_LAYER_HPP_
#define CAFFE_SIGMOID_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief the enhanced version of eltwise(op: SUM) with wighted 
 * Σi(wi * Inputi)
 */
template <typename Dtype>
class WightEltwiseLayer : public Layer<Dtype> {
 public:
  explicit WightEltwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  virtual inline const char* type() const { return "WightEltwise"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  WightedEltwiseParameter_FusionOp fusionOp_;
  Blob<Dtype> weight_Normal_;
  Blob<Dtype> temp_diff_;
  Blob<Dtype> temp_Fusion_data_;
};

}  // namespace caffe

#endif  // CAFFE_SWISH_LAYER_HPP_
