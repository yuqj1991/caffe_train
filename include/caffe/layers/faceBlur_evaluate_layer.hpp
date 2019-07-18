#ifndef CAFFE_DETECTION_EVALUATE_LAYER_HPP_
#define CAFFE_DETECTION_EVALUATE_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Generate the detection evaluation based on DetectionOutputLayer and
 *
 * Intended for use with multifaceBlur detection method.
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class faceBlurEvaluateLayer : public Layer<Dtype> {
 public:
  explicit faceBlurEvaluateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "faceBlurEvaluate"; }
  virtual inline int ExactBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
};

}  // namespace caffe

#endif  // CAFFE_DETECTION_EVALUATE_LAYER_HPP_
