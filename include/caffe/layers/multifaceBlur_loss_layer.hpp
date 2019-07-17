#ifndef CAFFE_MULTIBOX_LOSS_LAYER_HPP_
#define CAFFE_MULTIBOX_LOSS_LAYER_HPP_

#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Perform MultiBox operations. Including the following:
 *
 *  - decode the predictions.
 *  - perform matching between priors/predictions and ground truth.
 *  - use matched boxes and confidences to compute loss.
 *
 */
template <typename Dtype>
class MultiBlurLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiBlurLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiBlurLoss"; }
  // bottom[0] stores the location predictions.
  // bottom[1] stores the confidence predictions.
  // bottom[2] stores the prior bounding boxes.
  // bottom[3] stores the ground truth bounding boxes.
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // The internal confidence blur_loss_layer.
  shared_ptr<Layer<Dtype> > blur_loss_layer_;
  ConfLossType blur_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> blur_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> blur_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> blur_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> blur_gt_;
  // confidence loss.
  Blob<Dtype> blur_loss_;

  // The internal confidence occlu_loss_layer.
  shared_ptr<Layer<Dtype> > occlu_loss_layer_;
  ConfLossType occlu_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> occlu_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> occlu_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> occlu_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> occlu_gt_;
  // confidence loss.
  Blob<Dtype> occlu_loss_;

  MultiBoxLossParameter multibox_loss_param_;
  int num_blur_;
  int num_occlu_;
  int batch_size_;
  // How to normalize the loss.
  LossParameter_NormalizationMode normalization_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
