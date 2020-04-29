#ifndef CAFFE_SIGMOID_LAYER_HPP_
#define CAFFE_SIGMOID_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Swish function non-linearity @f$
 *         y = x * (1 + \exp(-beta* x))^{-1}
 *     @f$, a classic choice in neural networks.
 *
 * Note that the gradient vanishes as the values move away from 0.
 * The ReLULayer is often a better choice for this reason.
 */
template <typename Dtype>
class SwishLayer : public NeuronLayer<Dtype> {
 public:
  explicit SwishLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual inline const char* type() const { return "Swish"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_SWISH_LAYER_HPP_
