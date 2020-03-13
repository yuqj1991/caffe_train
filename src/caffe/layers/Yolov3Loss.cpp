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
  bottom_size_ = bottom.size();
  if (this->layer_param_.propagate_down_size() == 0) {
    for(int i = 0; i < bottom_size_ - 1; i++)
      this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
  }
  const CenterObjectParameter& center_object_loss_param =
      this->layer_param_.center_object_loss_param();
  
  if(center_object_loss_param.has_bias_num()){
    for(int i = 0; i < center_object_loss_param.bias_scale_size() / 2; i++){
      bias_scale_.push_back(std::pair<int, int>(center_object_loss_param.bias_scale(i * 2), 
                    center_object_loss_param.bias_scale(i * 2 + 1)));
    }
    bias_mask_group_num_ = center_object_loss_param.bias_mask_group_num();
    CHECK_EQ(bottom_size_, bias_mask_group_num_) << "bias_mask_group must be equal to bottom size";
    int num_mask_per_group = center_object_loss_param.bias_mask_size() / bias_mask_group_num_ ; 

    for(int j = 0; j < bias_mask_group_num_; j++){
      std::vector<int> mask_group_index;
      for(int i = 0; i < num_mask_per_group; i++){
        mask_group_index.clear();
        mask_group_index.push_back(center_object_loss_param.bias_mask(j * num_mask_per_group + i));
      }
      bias_mask_.push_back(std::pair<int, std::vector<int> >(j, mask_group_index));
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
  bottom_size_ = bottom.size();
  const Dtype* gt_data = bottom[bottom_size_ - 1]->cpu_data();
  num_gt_ = bottom[bottom_size_ - 1]->height(); 
  bool use_difficult_gt_ = true;
  Dtype background_label_id_ = -1;
  all_gt_bboxes.clear();
  GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                 &all_gt_bboxes);
  // prediction data
  for(unsigned i = 0; i < bottom_size_ - 1; i++){
    Dtype* channel_pred_data = bottom[i]->mutable_cpu_data();
    const int output_height = bottom[i]->height();
    const int output_width = bottom[i]->width();
    const int num_channels = bottom[i]->channels();
    Dtype * bottom_diff = bottom[i]->mutable_cpu_diff();

    num_ = bottom[i]->num();  

    caffe_set(bottom[i]->count(), Dtype(0), bottom_diff);
    if (num_gt_ >= 1) {
      EncodeYoloObject(num_, num_channels, num_classes_, output_width, output_height, 
                            net_width_, net_height_,
                            channel_pred_data, all_gt_bboxes,
                            bias_mask_[i].second, bias_scale_, 
                            bottom_diff, ignore_thresh_);
      const Dtype * diff = bottom[i]->cpu_diff();
      Dtype sum_squre = Dtype(0);
      for(int j = 0; j < bottom[i]->count(); j++){
        sum_squre += std::pow(diff[j], 2);
      }
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_gt_, num_gt_);
      top[0]->mutable_cpu_data()[0] += sum_squre / normalizer;
      
    } else {
      top[0]->mutable_cpu_data()[0] += 0;
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
}

template <typename Dtype>
void Yolov3LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[bottom_size_ - 1]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  for(unsigned i = 0; i < bottom_size_ - 1; i++){
    if (propagate_down[i]) {
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_gt_, num_gt_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
        caffe_scal(bottom[i]->count(), loss_weight, bottom[i]->mutable_cpu_diff());
    }
  }
}

INSTANTIATE_CLASS(Yolov3LossLayer);
REGISTER_LAYER_CLASS(Yolov3Loss);

}  // namespace caffe
