#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/CenternetLossLayer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterObjectLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    if (this->layer_param_.propagate_down_size() == 0) {
        this->layer_param_.add_propagate_down(true);
        this->layer_param_.add_propagate_down(true);
        this->layer_param_.add_propagate_down(false);
    }
    const CenterObjectLossParameter& center_object_loss_param =
        this->layer_param_.center_object_loss_param();

    num_classes_ = center_object_loss_param.num_class();
    CHECK_GE(num_classes_, 1) << "num_classes should not be less than 1.";
    CHECK_EQ(num_classes_, bottom[1]->channels()) << "num_classes must be equal to prediction classes";

    num_ = bottom[0]->num();
    num_gt_ = bottom[2]->height();

    share_location_ = center_object_loss_param.share_location();
    loc_classes_ = share_location_ ? 1 : num_classes_;

    if (!this->layer_param_.loss_param().has_normalization() &&
        this->layer_param_.loss_param().has_normalize()) {
        normalization_ = this->layer_param_.loss_param().normalize() ?
                        LossParameter_NormalizationMode_VALID :
                        LossParameter_NormalizationMode_BATCH_SIZE;
    } else {
        normalization_ = this->layer_param_.loss_param().normalization();
    }

    vector<int> loss_shape(1, 1);
    // Set up localization offset loss layer.
    loc_weight_ = center_object_loss_param.loc_weight();
    loc_loss_type_ = center_object_loss_param.loc_loss_type();
    // fake shape.
    vector<int> loc_shape(1, 1);
    loc_shape.push_back(4);
    loc_pred_.Reshape(loc_shape);
    loc_gt_.Reshape(loc_shape);
    //loc_channel_gt_.Reshape(loc_shape);
    loc_bottom_vec_.push_back(&loc_pred_);
    loc_bottom_vec_.push_back(&loc_gt_);
    //loc_bottom_vec_.push_back(&loc_channel_gt_);
    loc_loss_.Reshape(loss_shape);
    loc_top_vec_.push_back(&loc_loss_);
    if (loc_loss_type_ == CenterObjectLossParameter_LocLossType_L2) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_l2_loc");
        layer_param.set_type("EuclideanLoss");
        layer_param.add_loss_weight(loc_weight_);
        loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
    } else if (loc_loss_type_ == CenterObjectLossParameter_LocLossType_SMOOTH_L1) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
        layer_param.set_type("SmoothL1Loss");
        layer_param.add_loss_weight(loc_weight_);
        loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
    } else {
        LOG(FATAL) << "Unknown localization loss type.";
    }
    // Set up confidence loss layer.
    conf_loss_type_ = center_object_loss_param.conf_loss_type();
    conf_bottom_vec_.push_back(&conf_pred_);
    conf_bottom_vec_.push_back(&conf_gt_);
    conf_loss_.Reshape(loss_shape);
    conf_top_vec_.push_back(&conf_loss_);
    if (conf_loss_type_ == CenterObjectLossParameter_ConfLossType_FOCALSIGMOID) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_sigmoid_conf");
        layer_param.set_type("CenterNetfocalSigmoidWithLoss");
        layer_param.add_loss_weight(Dtype(1.));
        layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
        // Fake reshape.
        vector<int> conf_shape(1, 1);
        conf_gt_.Reshape(conf_shape);
        conf_shape.push_back(num_classes_);
        conf_pred_.Reshape(conf_shape);
        conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
    } else {
        LOG(FATAL) << "Unknown confidence loss type.";
    }
    iterations_ = 0;
}

template <typename Dtype>
void CenterObjectLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_GE(num_classes_, 1) << "num_classes should not be less than 1.";
    CHECK_EQ(num_classes_, bottom[1]->channels()) << "num_classes must be equal to prediction classes";
}

template <typename Dtype>
void CenterObjectLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* loc_data = bottom[0]->cpu_data();
    const int output_height = bottom[0]->height();
    const int output_width = bottom[0]->width();
    const int num_channels = bottom[0]->channels();
        
    const Dtype* gt_data = bottom[2]->cpu_data();
    num_gt_ = bottom[2]->height();

    // Retrieve all ground truth.
    bool use_difficult_gt_ = true;
    Dtype background_label_id_ = -1;
    all_gt_bboxes.clear();
    GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                    &all_gt_bboxes);
    int num_groundtruth = 0;
    for(int i = 0; i < all_gt_bboxes.size(); i++){
        vector<NormalizedBBox> gt_boxes = all_gt_bboxes[i];
        num_groundtruth += gt_boxes.size();
    }
    CHECK_EQ(num_gt_, num_groundtruth);
  
    if (num_gt_ >= 1) {
        // Form data to pass on to loc_loss_layer_.
        vector<int> loc_shape(2);
        loc_shape[0] = 1;
        loc_shape[1] = num_gt_ * 4;
        loc_pred_.Reshape(loc_shape);
        loc_gt_.Reshape(loc_shape);
        //loc_channel_gt_.Reshape(loc_shape);
        Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
        Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();
        /*
        Dtype* loc_channel_gt_data = loc_channel_gt_.mutable_cpu_data();
        for(int ii = 0; ii < num_gt_ * 4; ii++){
            int idx = ii % 4;
            loc_channel_gt_data[ii] = Dtype((idx < 2)? 1.f : 0.1f);
        }
        */
        EncodeTruthAndPredictions(loc_gt_data, loc_pred_data, 
                                            output_width, output_height, share_location_,
                                            loc_data, num_channels, all_gt_bboxes);
        loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
        loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);

    } else {
        loc_loss_.mutable_cpu_data()[0] = 0;
    }

    if (num_gt_ >= 1) {
        if (conf_loss_type_ == CenterObjectLossParameter_ConfLossType_FOCALSIGMOID) {
            conf_gt_.ReshapeLike(*bottom[1]);
            conf_pred_.ReshapeLike(*bottom[1]);
            conf_pred_.CopyFrom(*bottom[1]);
        }else {
            LOG(FATAL) << "Unknown confidence loss type.";
        }
        Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
        caffe_set(conf_gt_.count(), Dtype(0), conf_gt_data);
        GenerateBatchHeatmap(all_gt_bboxes, conf_gt_data, num_classes_, output_width, output_height);
        conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
        conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);
    } else {
        conf_loss_.mutable_cpu_data()[0] = 0;
    }

    top[0]->mutable_cpu_data()[0] = 0;
    Dtype loc_loss = Dtype(0.), cls_loss = Dtype(0.);
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
            normalization_, num_, 1, num_gt_);
    if (this->layer_param_.propagate_down(0)) {
        loc_loss = loc_weight_ * Dtype(loc_loss_.cpu_data()[0] / normalizer) ;
    }
    if (this->layer_param_.propagate_down(1)) {
        cls_loss = Dtype(conf_loss_.cpu_data()[0] / normalizer);
    }
    top[0]->mutable_cpu_data()[0] = cls_loss + loc_loss;
    #if 1 
    if(iterations_ % 100 == 0){
        LOG(INFO)<<"total loss: "<<top[0]->mutable_cpu_data()[0]
                <<", loc loss: "<<loc_loss
                <<", conf loss: "<< cls_loss
                <<", normalizer: "<< normalizer
                <<", num_classes: "<<num_classes_<<", output_width: "<<output_width
                <<", output_height: "<<output_height;
    }
    iterations_++;
    #endif
}

template <typename Dtype>
void CenterObjectLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const int output_height = bottom[0]->height();
    const int output_width = bottom[0]->width();
    const int num_channels = bottom[0]->channels();
    if (propagate_down[2]) {
        LOG(FATAL) << this->type()
            << " Layer cannot backpropagate to label inputs.";
    }

    // Back propagate on location offset prediction.
    if (propagate_down[0]) {
        Dtype* loc_bottom_diff = bottom[0]->mutable_cpu_diff();
        caffe_set(bottom[0]->count(), Dtype(0), loc_bottom_diff);
        if (num_gt_ >= 1) {
            vector<bool> loc_propagate_down;
            // Only back propagate on prediction, not ground truth.
            loc_propagate_down.push_back(true);
            loc_propagate_down.push_back(false);
            //loc_propagate_down.push_back(false);
            loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down,
                                        loc_bottom_vec_);
            Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
            normalization_, num_, 1, num_gt_);
            Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
            caffe_scal(loc_pred_.count(), loss_weight, loc_pred_.mutable_cpu_diff());
            const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
            CopyDiffToBottom(loc_pred_diff, output_width, output_height, 
                                share_location_, loc_bottom_diff, num_channels,
                                all_gt_bboxes);
        }
    }
    // Back propagate on confidence prediction.
    if (propagate_down[1]) {
        Dtype* conf_bottom_diff = bottom[1]->mutable_cpu_diff();
        caffe_set(bottom[1]->count(), Dtype(0), conf_bottom_diff);
        if (num_gt_ >= 1) {
            vector<bool> conf_propagate_down;
            conf_propagate_down.push_back(true);
            conf_propagate_down.push_back(false);
            conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down,
                                        conf_bottom_vec_);
            Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
            normalization_, num_, 1, num_gt_);
            Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
            caffe_scal(conf_pred_.count(), loss_weight, conf_pred_.mutable_cpu_diff());
            bottom[1]->ShareDiff(conf_pred_);
        }
    }
}

INSTANTIATE_CLASS(CenterObjectLossLayer);
REGISTER_LAYER_CLASS(CenterObjectLoss);

}  // namespace caffe
