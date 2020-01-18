#ifndef CAFFE_SAMPLE_TRIPLET_LAYER_HPP_
#define CAFFE_SAMPLE_TRIPLET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
 * triplet selection
 */
template <typename Dtype>
class SampleTripletLayer : public NeuronLayer<Dtype> {
 public:
  explicit SampleTripletLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SampleTriplet"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // do nothing
  }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  int triplet_num_;
  int sample_num_;
  int label_num_;
  int feature_dim_;
  int batch_size_;
  Blob<Dtype> inner_matrix_;
  vector< std::pair<int, float> > neg_dist_sqr;
  float alpha_;
  vector <int> an_set;
  vector <int> positive_set;
  vector <int> neg_set;
};

}  // namespace caffe

#endif  // CAFFE_SAMPLE_TRIPLET_LAYER_HPP_
