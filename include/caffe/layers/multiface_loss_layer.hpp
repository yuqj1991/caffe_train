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
class MultiFaceLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiFaceLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiFaceLoss"; }
  // bottom[0] stores the landmark predictions. 
  // bottom[1] stores the faceangle predictions
  // bottom[2] stores the gender predictions.
  // bottom[3] stores the glasses predictions.
  // bottom[4] stores the groundtruth labels.
  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // The internal landmark loss layer.
  shared_ptr<Layer<Dtype> > landmark_loss_layer_;
  MarkLossType landmark_loss_type_;
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

  // The internal confidence category loss layer.
  shared_ptr<Layer<Dtype> > gender_loss_layer_;
  AttriLossType gender_loss_type_;
  float gender_weight_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> gender_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> gender_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> gender_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> gender_gt_;
  // confidence loss.
  Blob<Dtype> gender_loss_;

  // The internal confidence category loss layer.
  shared_ptr<Layer<Dtype> > glasses_loss_layer_;
  AttriLossType glasses_loss_type_;
  float glasses_weight_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> glasses_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> glasses_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> glasses_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> glasses_gt_;
  // confidence loss.
  Blob<Dtype> glasses_loss_;

  // The internal category loss layer.
  shared_ptr<Layer<Dtype> > angle_loss_layer_;
  MarkLossType angle_loss_type_;
  float angle_weight_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> angle_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> angle_top_vec_;
  // blob which stores the  prediction.
  Blob<Dtype> angle_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> angle_gt_;
  //  loss.
  Blob<Dtype> angle_loss_;

  MultiFaceAttriLossParameter multiface_loss_param_;
  int num_gender_;
  int num_glasses_;
  int batch_size_;
  
  // How to normalize the loss.
  LossParameter_NormalizationMode normalization_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
