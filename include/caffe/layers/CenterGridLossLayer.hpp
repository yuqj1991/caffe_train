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
 * @brief 
 * Yolo Loss layer
 *
 */
template <typename Dtype>
class CenterGridLossLayer : public LossLayer<Dtype> {
 public:
  explicit CenterGridLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CenterGridLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_classes_;

  int num_gt_;
  int num_;
  int num_groundtruth_;


  std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes;

  int iterations_;

  // How to normalize the loss.
  LossParameter_NormalizationMode normalization_;

  std::pair<int, int>  bbox_range_scale_;
  int anchor_scale_;
  int net_width_;
  int net_height_;
  Dtype ignore_thresh_;
  Blob<Dtype> label_data_;
  int count_postive_;
  CenterObjectLossParameter_CLASS_TYPE class_type_;
  bool has_lm_;
  int num_lm_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
