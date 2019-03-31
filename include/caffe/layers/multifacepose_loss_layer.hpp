#ifndef CAFFE_MULTIFACE_LOSS_LAYER_HPP_
#define CAFFE_MULTIFACE_LOSS_LAYER_HPP_

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
 * @brief Perform Mutiface operations. Including the following:
 *
 *  
 *  - perform matching between priors/predictions and ground truth.
 *  - use matched landmarks and confidences to compute loss.
 *
 */
template <typename Dtype>
class MultiFacePoseLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiFacePoseLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiFacePoseLoss"; }
  // bottom[0] stores the 21 landmark face predictions.
  // bottom[1] stores the facepose contain yaw , pitch , raw predictions.
  // bottom[2] stores the groundtruth labels.
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // The internal landmark loss layer.
  shared_ptr<Layer<Dtype> > landmark_loss_layer_;
  AttriLossType landmark_loss_type_;
  float landmark_weight_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> landmark_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> landmark_top_vec_;
  // blob which stores the matched location prediction.
  Blob<Dtype> landmark_pred_;
  // blob which stores the corresponding matched ground truth.
  Blob<Dtype> landmark_gt_;
  // landmark loss.
  Blob<Dtype> landmark_loss_;

  // The internal poseface loss layer.
  shared_ptr<Layer<Dtype> > pose_loss_layer_;
  AttriLossType pose_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> pose_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>* pose_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> pose_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> pose_gt_;
  // confidence loss.
  Blob<Dtype> pose_loss_;

  MultiFacePoseLossParameter multiface_loss_param_;
  int batch_size_;
  
  // How to normalize the loss.
  LossParameter_NormalizationMode normalization_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
