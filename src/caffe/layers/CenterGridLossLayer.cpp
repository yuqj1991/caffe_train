#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/CenterGridLossLayer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterGridLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    if (this->layer_param_.propagate_down_size() == 0) {
        this->layer_param_.add_propagate_down(true);
        this->layer_param_.add_propagate_down(false);
    }
    const CenterObjectLossParameter& center_object_loss_param =
        this->layer_param_.center_object_loss_param();
    
    // bias_mask_
    CHECK_EQ(center_object_loss_param.has_bias_num(), 1);
    if(center_object_loss_param.has_bias_num()){
        CHECK_EQ(center_object_loss_param.bias_scale_size(), 1);
        CHECK_EQ(center_object_loss_param.bias_num(), 1);
        anchor_scale_ = center_object_loss_param.bias_scale(0);
        int low_bbox = center_object_loss_param.low_bbox_scale();
        int up_bbox = center_object_loss_param.up_bbox_scale();
        bbox_range_scale_ = std::make_pair(low_bbox, up_bbox);
    }
  
    net_height_ = center_object_loss_param.net_height();
    net_width_ = center_object_loss_param.net_width();
    ignore_thresh_ = center_object_loss_param.ignore_thresh();

    has_lm_ = center_object_loss_param.has_lm();
    
    if (!this->layer_param_.loss_param().has_normalization() &&
        this->layer_param_.loss_param().has_normalize()) {
        normalization_ = this->layer_param_.loss_param().normalize() ?
                        LossParameter_NormalizationMode_VALID :
                        LossParameter_NormalizationMode_BATCH_SIZE;
    } else {
        normalization_ = this->layer_param_.loss_param().normalization();
    }
    iterations_ = 0;
    vector<int> label_shape(1, 1);
    label_shape.push_back(1);
    label_data_.Reshape(label_shape);
    class_type_ = center_object_loss_param.class_type();
    num_classes_ = center_object_loss_param.num_class();
    CHECK_GE(num_classes_, 1) << "num_classes should not be less than 1.";
    if(class_type_ == CenterObjectLossParameter_CLASS_TYPE_SIGMOID){
        CHECK_GE(num_classes_, 1)
                << "num_classes must be more than equal to prediction classes";
    }else if(class_type_ == CenterObjectLossParameter_CLASS_TYPE_SOFTMAX){
        CHECK_GT(num_classes_, 1)
                << "softmax num_classes must be equal to contain background";
    }else{
        LOG(FATAL)<<"unknown class type";
    }

    CHECK_EQ(center_object_loss_param.share_location(), true);
    normalized_changed_ = false;
}

template <typename Dtype>
void CenterGridLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void CenterGridLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
    // gt_boxes
    const Dtype* gt_data = bottom[1]->cpu_data();
    num_gt_ = bottom[1]->height(); 
    bool use_difficult_gt_ = true;
    Dtype background_label_id_ = -1;
    num_ = bottom[0]->num();
    all_gt_bboxes.clear();
    
    GetYoloGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_, has_lm_,
                    &all_gt_bboxes, num_);
    num_groundtruth_ = 0;
    num_lm_ = 0;
    for(int i = 0; i < all_gt_bboxes.size(); i++){
        vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > gt_boxes = all_gt_bboxes[i];
        num_groundtruth_ += gt_boxes.size();
        for(unsigned ii = 0;  ii < gt_boxes.size(); ii++){
            if(gt_boxes[ii].second.lefteye().x() > 0 && gt_boxes[ii].second.lefteye().y() > 0 &&
                   gt_boxes[ii].second.righteye().x() > 0 && gt_boxes[ii].second.righteye().y() > 0 && 
                   gt_boxes[ii].second.nose().x() > 0 && gt_boxes[ii].second.nose().y() > 0 &&
                   gt_boxes[ii].second.leftmouth().x() > 0 && gt_boxes[ii].second.leftmouth().y() > 0 &&
                   gt_boxes[ii].second.rightmouth().x() > 0 && gt_boxes[ii].second.rightmouth().y() > 0){
                num_lm_++;
            }
        }
    }
    // prediction data
    Dtype* channel_pred_data = bottom[0]->mutable_cpu_data();
    const int output_height = bottom[0]->height();
    const int output_width = bottom[0]->width();
    const int num_channels = bottom[0]->channels();
    Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();

    std::vector<int> postive_batch_(num_, 0);
    //计算每个样本的总损失（loc loss + softmax loss)
    std::vector<Dtype> batch_sample_loss_(num_ * output_height * output_width, Dtype(-1.));
    
    vector<int> label_shape(2, 1);
    label_shape.push_back(num_);
    label_shape.push_back(output_height*output_width);
    label_data_.Reshape(label_shape);

    Dtype *label_muti_data = label_data_.mutable_cpu_data();
    Dtype class_score = Dtype(0.);
    Dtype sum_squre = Dtype(0.);
    Dtype lm_loss_origin = Dtype(0.);
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    Dtype loc_loss = Dtype(0.), score_loss = Dtype(0.), normalizer = Dtype(0.), lm_loss = Dtype(0.), 
                    lm_normalizer = Dtype(0.);
    int num_gt_match = 0;
    if (num_groundtruth_ >= 1) {
        const float downRatio = (float)net_height_ / output_height;
        if(class_type_ == CenterObjectLossParameter_CLASS_TYPE_SIGMOID){
            class_score = EncodeCenterGridObjectSigmoidLoss(num_, num_channels, num_classes_, output_width, output_height, 
                            downRatio,
                            channel_pred_data,  anchor_scale_, 
                            bbox_range_scale_,
                            all_gt_bboxes, label_muti_data, bottom_diff, 
                            ignore_thresh_, &count_postive_, &sum_squre);
        }else if(class_type_ == CenterObjectLossParameter_CLASS_TYPE_SOFTMAX){
            class_score = EncodeCenterGridObjectSoftMaxLoss(num_, num_channels, num_classes_, output_width, output_height, 
                            downRatio, postive_batch_, batch_sample_loss_,
                            channel_pred_data,  anchor_scale_, 
                            bbox_range_scale_,
                            all_gt_bboxes, label_muti_data, bottom_diff, &count_postive_lm_,
                            &count_postive_, &sum_squre, &num_gt_match, has_lm_, &lm_loss_origin);
        }
        normalizer = LossLayer<Dtype>::GetNormalizer(
                                        normalization_, num_, 1, count_postive_);
        lm_normalizer = LossLayer<Dtype>::GetNormalizer(
                                        normalization_, num_, 1, count_postive_lm_*10);
        loc_loss = sum_squre / normalizer;
        score_loss = class_score / normalizer;
        top[0]->mutable_cpu_data()[0] = loc_loss + score_loss;
        if(has_lm_ && num_lm_ > 0){
            lm_loss = 0.1 * lm_loss_origin / lm_normalizer;
            top[0]->mutable_cpu_data()[0] += lm_loss;
        }
    } else {
        top[0]->mutable_cpu_data()[0] = 0;
    }
    #if 1
    if(iterations_ % 100 == 0){
        LOG(INFO)<<"Region "<<output_width
                <<": anchor scale: "<<anchor_scale_
                <<", total loss: "<<top[0]->mutable_cpu_data()[0]
                <<", loc loss: "<< loc_loss
                <<", class loss: "<< score_loss
                <<", lm_loss: "<<lm_loss;
        LOG(INFO)<<"all_gt_boxes: "<<num_groundtruth_
                <<", normalizer: "<<normalizer
                <<", count_postive: "<< count_postive_
                <<", num_match: "<<num_gt_match
                <<", num_landmarks: "<<num_lm_;
        LOG(INFO)<<"*****************";
                
    }
    iterations_++;
    #endif
}

template <typename Dtype>
void CenterGridLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << this->type()
            << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
        Dtype loss_weight = Dtype(0.);
        Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
                                        normalization_, num_, 1, count_postive_);
        Dtype lm_normalizer = LossLayer<Dtype>::GetNormalizer(
                                        normalization_, num_, 1, count_postive_lm_*10);
        
        const int output_height = bottom[0]->height();
        const int output_width = bottom[0]->width();
        const int num_channels = bottom[0]->channels();
        int spatial_dim = output_height * output_width;

        loss_weight = top[0]->cpu_diff()[0] / normalizer;
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        for(int i = 0; i < num_; i++){
            caffe_cpu_scale(4 * spatial_dim, loss_weight, 
                                bottom_diff + i * num_channels * spatial_dim, 
                                bottom_diff + i * num_channels * spatial_dim);
        }
        if(has_lm_){
            loss_weight = top[0]->cpu_diff()[0] / lm_normalizer;
            for(int i = 0; i < num_; i++)
                caffe_cpu_scale(10 * spatial_dim, Dtype(0.1) * loss_weight, 
                    bottom_diff + i * num_channels * spatial_dim + 4 * spatial_dim, 
                    bottom_diff + i * num_channels * spatial_dim + 4 * spatial_dim);
        }
    }
}

INSTANTIATE_CLASS(CenterGridLossLayer);
REGISTER_LAYER_CLASS(CenterGridLoss);

}  // namespace caffe
