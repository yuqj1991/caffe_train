#ifndef CAFFE_MULTIBOX_LOSS_LAYER_HPP_CENTERNET_OBJECT_
#define CAFFE_MULTIBOX_LOSS_LAYER_HPP_CENTERNET_OBJECT_

#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/center_bbox_util.hpp"
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
class CenterObjectLossLayer : public LossLayer<Dtype> {
 public:
  explicit CenterObjectLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CenterObjectLoss"; }
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

  // The internal localization offset loss layer.
  shared_ptr<Layer<Dtype> > loc_loss_layer_;
  CenterObjectLossParameter_LocLossType loc_loss_type_;
  float loc_weight_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> loc_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> loc_top_vec_;
  // blob which stores the matched location prediction.
  Blob<Dtype> loc_pred_;
  // blob which stores the corresponding matched ground truth.
  Blob<Dtype> loc_gt_;
  // blob loc_loss_channel with weight
  Blob<Dtype> loc_channel_gt_;
  // localization loss.
  Blob<Dtype> loc_loss_;

#if 0
  // The internal  object scale loss layer.
  shared_ptr<Layer<Dtype> > wh_loss_layer_;
  CenterObjectLossParameter_LocLossType wh_loss_type_;
  float wh_weight_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> wh_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> wh_top_vec_;
  // blob which stores the matched location prediction.
  Blob<Dtype> wh_pred_;
  // blob which stores the corresponding matched ground truth.
  Blob<Dtype> wh_gt_;
  // localization loss.
  Blob<Dtype> wh_loss_;
#endif
  // The internal confidence loss layer.
  shared_ptr<Layer<Dtype> > conf_loss_layer_;
  CenterObjectLossParameter_ConfLossType conf_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> conf_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> conf_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> conf_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> conf_gt_;
  // confidence loss.
  Blob<Dtype> conf_loss_;

  int num_classes_;
  bool share_location_;

  CodeType code_type_;

  int loc_classes_;
  int num_gt_;
  int num_;


  std::map<int, vector<NormalizedBBox> > all_gt_bboxes;

  int iterations_;

  // How to normalize the loss.
  LossParameter_NormalizationMode normalization_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
