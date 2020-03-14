#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/Yolov3Loss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Yolov3LossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
  }
  const CenterObjectParameter& center_object_loss_param =
      this->layer_param_.center_object_loss_param();
  
  // bias_mask_
  if(center_object_loss_param.has_bias_num()){
    for(int i = 0; i < center_object_loss_param.bias_scale_size() / 2; i++){
      bias_scale_.push_back(std::pair<int, int>(center_object_loss_param.bias_scale(i * 2), 
                    center_object_loss_param.bias_scale(i * 2 + 1)));
    }
    for(int i = 0; i < center_object_loss_param.bias_mask_size(); i++){
      bias_mask_.push_back(center_object_loss_param.bias_mask(i));
    }
  }
  
  net_height_ = center_object_loss_param.net_height();
  net_width_ = center_object_loss_param.net_width();
  bias_num_ = center_object_loss_param.bias_num();
  ignore_thresh_ = center_object_loss_param.ignore_thresh();
  CHECK_EQ(bias_num_, bias_scale_.size()); // anchor size
  
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
void Yolov3LossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void Yolov3LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  // gt_boxes
  const Dtype* gt_data = bottom[1]->cpu_data();
  num_gt_ = bottom[1]->height(); 
  bool use_difficult_gt_ = true;
  Dtype background_label_id_ = -1;
  num_ = bottom[0]->num();
  all_gt_bboxes.clear();
  
  GetYoloGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                 &all_gt_bboxes, num_);
  num_groundtruth_ = 0;
  for(int i = 0; i < all_gt_bboxes.size(); i++){
    vector<NormalizedBBox> gt_boxes = all_gt_bboxes[i];
    num_groundtruth_ += gt_boxes.size();
  }
  // prediction data
  Dtype* channel_pred_data = bottom[0]->mutable_cpu_data();
  const int output_height = bottom[0]->height();
  const int output_width = bottom[0]->width();
  const int num_channels = bottom[0]->channels();
  Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();

  YoloScoreShow trainScore;
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  if (num_groundtruth_ >= 1) {
    EncodeYoloObject(num_, num_channels, num_classes_, output_width, output_height, 
                          net_width_, net_height_,
                          channel_pred_data, all_gt_bboxes,
                          bias_mask_, bias_scale_, 
                          bottom_diff, ignore_thresh_, &trainScore);
    const Dtype * diff = bottom[0]->cpu_diff();
    Dtype sum_squre = Dtype(0.);
    for(int j = 0; j < bottom[0]->count(); j++){
      sum_squre += diff[j] * diff[j];
    }
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_groundtruth_, num_groundtruth_);
    top[0]->mutable_cpu_data()[0] += sum_squre / normalizer;
  } else {
    top[0]->mutable_cpu_data()[0] += 0;
  }
  #if 1 
  if(iterations_ % 50 == 0){    
    int dimScale = output_height * output_width;  
    LOG(INFO)<<"all num_gt boxes: "<<num_gt_;     
    LOG(INFO)<<"Region "<<output_width<<": total loss: "<<top[0]->mutable_cpu_data()[0]<<", num_groundtruth: "<<num_groundtruth_<<" Avg IOU: "
                      <<trainScore.avg_iou/trainScore.count<<", Class: "<<trainScore.avg_cat/trainScore.class_count
                      <<", Obj: "<<trainScore.avg_obj/trainScore.count<<", No obj: "<<trainScore.avg_anyobj/(dimScale*bias_mask_.size()*num_)
                      <<", .5R: "<<trainScore.recall/trainScore.count<<", .75R: "<<trainScore.recall75/trainScore.count
                      <<", count: "<<trainScore.count;
  }
  iterations_++;
  #endif
}

template <typename Dtype>
void Yolov3LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  
  if (propagate_down[0]) {
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_groundtruth_, num_groundtruth_);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    const int output_height = bottom[0]->height();
    const int output_width = bottom[0]->width();
    const int num_channels = bottom[0]->channels();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    num_ = bottom[0]->num();
    int channel_per_box = 4 + 1 + num_classes_;
    int mask_size_ = bias_mask_.size();
    int dimScale = output_height * output_width;
    CHECK_EQ(channel_per_box * mask_size_, num_channels);
    for(int j = 0; j < num_; j++){
      for(int mm = 0; mm < mask_size_; mm++){
        for(int cc = 0; cc < channel_per_box; cc++){
          int channal_index = j * num_channels * dimScale + (mm * channel_per_box + cc) * dimScale;
          if(cc != 2 && cc != 3){
            for(int s = 0; s < dimScale; s++){
              int index = channal_index + s;
              bottom_diff[index] = bottom_diff[index] * logistic_gradient(bottom_data[index]);
            }
          }
        }
      }
    } 
    caffe_scal(bottom[0]->count(), loss_weight, bottom[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(Yolov3LossLayer);
REGISTER_LAYER_CLASS(Yolov3Loss);

}  // namespace caffe
