#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/featuremap_bbox_object_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeaturemapObjectLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
  }
  const CenterObjectParameter& center_object_loss_param =
      this->layer_param_.center_object_loss_param();
  
  if(center_object_loss_param.has_bias_num()){
    for(int i = 0; i < center_object_loss_param.bias_scale_size(); i++){
      bias_scale.push_back(center_object_loss_param.bias_scale(i));
    }
    for(int i = 0; i < center_object_loss_param.bias_mask_size(); i++){
      bias_mask_.push_back(center_object_loss_param.bias_mask(i));
    }
  }
  net_height_ = center_object_loss_param.net_height();
  net_width_ = center_object_loss_param.net_width();
  bias_num_ = center_object_loss_param.bias_num();
  ignore_thresh_ = center_object_loss_param.ignore_thresh();
  CHECK_EQ(bias_num_, bias_scale_.size() / 2); // anchor size
  
  num_classes_ = center_object_loss_param.num_class();
  CHECK_GE(num_classes_, 1) << "num_classes should not be less than 1.";
  CHECK_EQ((4 + 1 + num_classes_) * bias_mask_.size(), bottom[0]->channels()) 
            << "num_classes must be equal to prediction classes";
  
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  iterations_ = 0;
}

template <typename Dtype>
void FeaturemapObjectLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void FeaturemapObjectLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* channel_pred_data = bottom[0]->cpu_data();
  const int output_height = bottom[0]->height();
  const int output_width = bottom[0]->width();
  const int num_channels = bottom[0]->channels();
  Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();

  num_ = bottom[0]->num();
  num_gt_ = bottom[1]->height(); 
  
  
  const Dtype* gt_data = bottom[1]->cpu_data();
  num_gt_ = bottom[1]->height(); 

  // Retrieve all ground truth.
  bool use_difficult_gt_ = true;
  Dtype background_label_id_ = -1;
  all_gt_bboxes.clear();
  GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                 &all_gt_bboxes);
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  if (num_gt_ >= 1) {
    EncodeYoloObject(num_, num_channels, num_classes_, output_width, output_height, 
                          net_width_, net_height_,
                          channel_pred_data, all_gt_bboxes,
                          bias_mask_, bias_scale_, 
                          bottom_diff, ignore_thresh_);
    const Dtype * diff = bottom[0]->cpu_diff();
    Dtype sum_squre = Dtype(0);
    for(int i = 0; i < bottom[0]->count(); i++){
      sum_squre += std::pow(diff[i], 2);
    }
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_gt_, num_gt_);
    top[0]->mutable_cpu_data()[0] = sum_squre / normalizer;
    
  } else {
    top[0]->mutable_cpu_data()[0] = 0;
  }

  #if 1 
  if(iterations_ % 1000 == 0){
    int num_groundtruth = 0;
    for(int i = 0; i < all_gt_bboxes.size(); i++){
      vector<NormalizedBBox> gt_boxes = all_gt_bboxes[i];
      num_groundtruth += gt_boxes.size();
    }
    CHECK_EQ(num_gt_, num_groundtruth);
    LOG(INFO)<<"total loss: "<<top[0]->mutable_cpu_data()[0]
            <<", num_groundtruth: "<<num_groundtruth
            <<", num_classes: "<<num_classes_<<", output_width: "<<output_width
            <<", output_height: "<<output_height;
  }
  iterations_++;
  #endif
}

template <typename Dtype>
void FeaturemapObjectLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_gt_, num_gt_);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(bottom[0]->count(), loss_weight, bottom[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(FeaturemapObjectLossLayer);
REGISTER_LAYER_CLASS(FeaturemapObjectLoss);

}  // namespace caffe
