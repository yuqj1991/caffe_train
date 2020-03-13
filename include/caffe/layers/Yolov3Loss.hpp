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
class Yolov3LossLayer : public LossLayer<Dtype> {
 public:
  explicit Yolov3LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Yolov3Loss"; }
  virtual inline int MinBottomBlobs() const { return 2; } 
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_classes_;

  int num_gt_;
  int num_;
  unsigned bottom_size_;


  std::map<int, vector<NormalizedBBox> > all_gt_bboxes;

  int iterations_;

  // How to normalize the loss.
  LossParameter_NormalizationMode normalization_;

  std::vector<std::pair<int, int> > bias_scale_;
  std::vector<std::pair<int, std::vector<int> > > bias_mask_;
  int bias_num_;
  int net_width_;
  int net_height_;
  Dtype ignore_thresh_;
  int bias_mask_group_num_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
