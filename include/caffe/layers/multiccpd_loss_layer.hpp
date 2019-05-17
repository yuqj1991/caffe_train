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
class MulticcpdLossLayer : public LossLayer<Dtype> {
 public:
  explicit MulticcpdLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MulticcpdLoss"; }
  // bottom[0] stores the location predictions.
  // bottom[1] stores the confidence predictions.
  // bottom[2] stores the prior bounding boxes.
  // bottom[3] stores the ground truth bounding boxes.
  virtual inline int ExactNumBottomBlobs() const { return 8; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // The internal confidence chinesecharcter_loss_layer.
  shared_ptr<Layer<Dtype> > chinesecharcter_loss_layer_;
  ConfLossType chinesecharcter_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> chinesecharcter_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> chinesecharcter_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> chinesecharcter_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> chinesecharcter_gt_;
  // confidence loss.
  Blob<Dtype> chinesecharcter_loss_;

  // The internal confidence engcharcter_loss_layer.
  shared_ptr<Layer<Dtype> > engcharcter_loss_layer_;
  ConfLossType engcharcter_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> engcharcter_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> engcharcter_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> engcharcter_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> engcharcter_gt_;
  // confidence loss.
  Blob<Dtype> engcharcter_loss_;

  // The internal confidence letternum_loss_layer.
  shared_ptr<Layer<Dtype> > letternum_1_loss_layer_;
  ConfLossType letternum_1_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> letternum_1_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> letternum_1_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> letternum_1_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> letternum_1_gt_;
  // confidence loss.
  Blob<Dtype> letternum_1_loss_;

  // The internal confidence letternum_loss_layer.
  shared_ptr<Layer<Dtype> > letternum_2_loss_layer_;
  ConfLossType letternum_2_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> letternum_2_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> letternum_2_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> letternum_2_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> letternum_2_gt_;
  // confidence loss.
  Blob<Dtype> letternum_2_loss_;

  // The internal confidence letternum_loss_layer.
  shared_ptr<Layer<Dtype> > letternum_3_loss_layer_;
  ConfLossType letternum_3_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> letternum_3_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> letternum_3_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> letternum_3_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> letternum_3_gt_;
  // confidence loss.
  Blob<Dtype> letternum_3_loss_;

  // The internal confidence letternum_loss_layer.
  shared_ptr<Layer<Dtype> > letternum_4_loss_layer_;
  ConfLossType letternum_4_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> letternum_4_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> letternum_4_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> letternum_4_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> letternum_4_gt_;
  // confidence loss.
  Blob<Dtype> letternum_4_loss_;

  // The internal confidence letternum_loss_layer.
  shared_ptr<Layer<Dtype> > letternum_5_loss_layer_;
  ConfLossType letternum_5_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> letternum_5_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> letternum_5_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> letternum_5_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> letternum_5_gt_;
  // confidence loss.
  Blob<Dtype> letternum_5_loss_;

  MultiBoxLossParameter multibox_loss_param_;
  int num_chinese_;
  int num_english_;
  int num_letter_;
  int batch_size_;

  // How to normalize the loss.
  LossParameter_NormalizationMode normalization_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
