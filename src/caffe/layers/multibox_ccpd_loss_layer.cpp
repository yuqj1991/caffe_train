#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multibox_ccpd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiBoxccpdLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_cnt_ = 0;
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
  }
  const MultiBoxLossParameter& multibox_loss_param =
      this->layer_param_.multibox_loss_param();
  multibox_loss_param_ = this->layer_param_.multibox_loss_param();

  num_ = bottom[0]->num();
  num_priors_ = bottom[2]->height() / 4;
  // Get other parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  CHECK(multibox_loss_param.has_num_chinese()) << "Must prodived num_chinese";
  CHECK(multibox_loss_param.has_num_english()) << "Must provide num_english";
  num_classes_ = multibox_loss_param.num_classes();
  num_chinese_ = multibox_loss_param.num_chinese();
  num_english_ = multibox_loss_param.num_english();
  num_letter_ = multibox_loss_param.num_letter();
  CHECK_GE(num_classes_, 1) << "num_classes should not be less than 1.";
  share_location_ = multibox_loss_param.share_location();
  loc_classes_ = share_location_ ? 1 : num_classes_;
  background_label_id_ = multibox_loss_param.background_label_id();
  use_difficult_gt_ = multibox_loss_param.use_difficult_gt();
  mining_type_ = multibox_loss_param.mining_type();
  if (multibox_loss_param.has_do_neg_mining()) {
    LOG(WARNING) << "do_neg_mining is deprecated, use mining_type instead.";
    do_neg_mining_ = multibox_loss_param.do_neg_mining();
    CHECK_EQ(do_neg_mining_,
             mining_type_ != MultiBoxLossParameter_MiningType_NONE);
  }
  do_neg_mining_ = mining_type_ != MultiBoxLossParameter_MiningType_NONE;

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  if (do_neg_mining_) {
    CHECK(share_location_)
        << "Currently only support negative mining if share_location is true.";
  }

  vector<int> loss_shape(1, 1);
  // Set up localization loss layer.
  loc_weight_ = multibox_loss_param.loc_weight();
  loc_loss_type_ = multibox_loss_param.loc_loss_type();
  // fake shape.
  vector<int> loc_shape(1, 1);
  loc_shape.push_back(4); //loc_shape:{ 1, 4}
  loc_pred_.Reshape(loc_shape); //loc_shape:{ }
  loc_gt_.Reshape(loc_shape);
  loc_bottom_vec_.push_back(&loc_pred_);
  loc_bottom_vec_.push_back(&loc_gt_);
  loc_loss_.Reshape(loss_shape);
  loc_top_vec_.push_back(&loc_loss_);
  if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_L2) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_l2_loc");
    layer_param.set_type("EuclideanLoss");
    layer_param.add_loss_weight(loc_weight_);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
  } else if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_SMOOTH_L1) {
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
  conf_loss_type_ = multibox_loss_param.conf_loss_type();
  conf_bottom_vec_.push_back(&conf_pred_);
  conf_bottom_vec_.push_back(&conf_gt_);
  conf_loss_.Reshape(loss_shape);
  conf_top_vec_.push_back(&conf_loss_);
  if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    CHECK_GE(background_label_id_, 0)
        << "background_label_id should be within [0, num_classes) for Softmax.";
    CHECK_LT(background_label_id_, num_classes_)
        << "background_label_id should be within [0, num_classes) for Softmax.";
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> conf_shape(1, 1);
    conf_gt_.Reshape(conf_shape);
    conf_shape.push_back(num_classes_);
    conf_pred_.Reshape(conf_shape);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
  } else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> conf_shape(1, 1);
    conf_shape.push_back(num_classes_);
    conf_gt_.Reshape(conf_shape);
    conf_pred_.Reshape(conf_shape);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  // Set up chinese confidence loss layer.
  chinesecharcter_loss_type_ = multibox_loss_param.chineselp_loss_type();
  chinesecharcter_bottom_vec_.push_back(&chinesecharcter_pred_);
  chinesecharcter_bottom_vec_.push_back(&chinesecharcter_gt_);
  chinesecharcter_loss_.Reshape(loss_shape);
  chinesecharcter_top_vec_.push_back(&chinesecharcter_loss_);
  if (chinesecharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_chinese_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> chinesecharcter_shape(1, 1);
    chinesecharcter_gt_.Reshape(chinesecharcter_shape);
    chinesecharcter_shape.push_back(num_chinese_);
    chinesecharcter_pred_.Reshape(chinesecharcter_shape);
    chinesecharcter_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    chinesecharcter_loss_layer_->SetUp(chinesecharcter_bottom_vec_, chinesecharcter_top_vec_);
  } else if (chinesecharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_chinese_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> chinesecharcter_shape(1, 1);
    chinesecharcter_shape.push_back(num_chinese_);
    chinesecharcter_gt_.Reshape(chinesecharcter_shape);
    chinesecharcter_pred_.Reshape(chinesecharcter_shape);
    chinesecharcter_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    chinesecharcter_loss_layer_->SetUp(chinesecharcter_bottom_vec_, chinesecharcter_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  // Set up engcharcter confidence loss layer.
  engcharcter_loss_type_ = multibox_loss_param.englishlp_loss_type();
  engcharcter_bottom_vec_.push_back(&engcharcter_pred_);
  engcharcter_bottom_vec_.push_back(&engcharcter_gt_);
  engcharcter_loss_.Reshape(loss_shape);
  engcharcter_top_vec_.push_back(&engcharcter_loss_);
  if (engcharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_english_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> engcharcter_shape(1, 1);
    engcharcter_gt_.Reshape(engcharcter_shape);
    engcharcter_shape.push_back(num_english_);
    engcharcter_pred_.Reshape(engcharcter_shape);
    engcharcter_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    engcharcter_loss_layer_->SetUp(engcharcter_bottom_vec_, engcharcter_top_vec_);
  } else if (engcharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_english_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> engcharcter_shape(1, 1);
    engcharcter_shape.push_back(num_english_);
    engcharcter_gt_.Reshape(engcharcter_shape);
    engcharcter_pred_.Reshape(engcharcter_shape);
    engcharcter_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    engcharcter_loss_layer_->SetUp(engcharcter_bottom_vec_, engcharcter_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }

  // Set up letternum confidence loss layer.
  letternum_1_loss_type_ = multibox_loss_param.letter_lp_loss_type();
  letternum_1_bottom_vec_.push_back(&letternum_1_pred_);
  letternum_1_bottom_vec_.push_back(&letternum_1_gt_);
  letternum_1_loss_.Reshape(loss_shape);
  letternum_1_top_vec_.push_back(&letternum_1_loss_);
  if (letternum_1_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_letter_1_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> letternum_1_shape(1, 1);
    letternum_1_gt_.Reshape(letternum_1_shape);
    letternum_1_shape.push_back(num_letter_);
    letternum_1_pred_.Reshape(letternum_1_shape);
    letternum_1_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    letternum_1_loss_layer_->SetUp(letternum_1_bottom_vec_, letternum_1_top_vec_);
  } else if (letternum_1_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_letter_1_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> letternum_1_shape(1, 1);
    letternum_1_shape.push_back(num_letter_);
    letternum_1_gt_.Reshape(letternum_1_shape);
    letternum_1_pred_.Reshape(letternum_1_shape);
    letternum_1_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    letternum_1_loss_layer_->SetUp(letternum_1_bottom_vec_, letternum_1_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }

  // Set up letternum confidence loss layer.
  letternum_2_loss_type_ = multibox_loss_param.letter_lp_loss_type();
  letternum_2_bottom_vec_.push_back(&letternum_2_pred_);
  letternum_2_bottom_vec_.push_back(&letternum_2_gt_);
  letternum_2_loss_.Reshape(loss_shape);
  letternum_2_top_vec_.push_back(&letternum_2_loss_);
  if (letternum_2_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_letter_2_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> letternum_2_shape(1, 1);
    letternum_2_gt_.Reshape(letternum_2_shape);
    letternum_2_shape.push_back(num_letter_);
    letternum_2_pred_.Reshape(letternum_2_shape);
    letternum_2_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    letternum_2_loss_layer_->SetUp(letternum_2_bottom_vec_, letternum_2_top_vec_);
  } else if (letternum_2_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_letter_2_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> letternum_2_shape(1, 1);
    letternum_2_shape.push_back(num_letter_);
    letternum_2_gt_.Reshape(letternum_2_shape);
    letternum_2_pred_.Reshape(letternum_2_shape);
    letternum_2_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    letternum_2_loss_layer_->SetUp(letternum_2_bottom_vec_, letternum_2_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }

  // Set up letternum confidence loss layer.
  letternum_3_loss_type_ = multibox_loss_param.letter_lp_loss_type();
  letternum_3_bottom_vec_.push_back(&letternum_3_pred_);
  letternum_3_bottom_vec_.push_back(&letternum_3_gt_);
  letternum_3_loss_.Reshape(loss_shape);
  letternum_3_top_vec_.push_back(&letternum_3_loss_);
  if (letternum_3_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_letter_3_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> letternum_3_shape(1, 1);
    letternum_3_gt_.Reshape(letternum_3_shape);
    letternum_3_shape.push_back(num_letter_);
    letternum_3_pred_.Reshape(letternum_3_shape);
    letternum_3_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    letternum_3_loss_layer_->SetUp(letternum_3_bottom_vec_, letternum_3_top_vec_);
  } else if (letternum_3_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_letter_3_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> letternum_3_shape(1, 1);
    letternum_3_shape.push_back(num_letter_);
    letternum_3_gt_.Reshape(letternum_3_shape);
    letternum_3_pred_.Reshape(letternum_3_shape);
    letternum_3_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    letternum_3_loss_layer_->SetUp(letternum_3_bottom_vec_, letternum_3_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }

  // Set up letternum confidence loss layer.
  letternum_4_loss_type_ = multibox_loss_param.letter_lp_loss_type();
  letternum_4_bottom_vec_.push_back(&letternum_4_pred_);
  letternum_4_bottom_vec_.push_back(&letternum_4_gt_);
  letternum_4_loss_.Reshape(loss_shape);
  letternum_4_top_vec_.push_back(&letternum_4_loss_);
  if (letternum_4_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_letter_4_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> letternum_4_shape(1, 1);
    letternum_4_gt_.Reshape(letternum_4_shape);
    letternum_4_shape.push_back(num_letter_);
    letternum_4_pred_.Reshape(letternum_4_shape);
    letternum_4_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    letternum_4_loss_layer_->SetUp(letternum_4_bottom_vec_, letternum_4_top_vec_);
  } else if (letternum_4_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_letter_4_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> letternum_4_shape(1, 1);
    letternum_4_shape.push_back(num_letter_);
    letternum_4_gt_.Reshape(letternum_4_shape);
    letternum_4_pred_.Reshape(letternum_4_shape);
    letternum_4_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    letternum_4_loss_layer_->SetUp(letternum_4_bottom_vec_, letternum_4_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }

  // Set up letternum confidence loss layer.
  letternum_5_loss_type_ = multibox_loss_param.letter_lp_loss_type();
  letternum_5_bottom_vec_.push_back(&letternum_5_pred_);
  letternum_5_bottom_vec_.push_back(&letternum_5_gt_);
  letternum_5_loss_.Reshape(loss_shape);
  letternum_5_top_vec_.push_back(&letternum_5_loss_);
  if (letternum_5_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_letter_5_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> letternum_5_shape(1, 1);
    letternum_5_gt_.Reshape(letternum_5_shape);
    letternum_5_shape.push_back(num_letter_);
    letternum_5_pred_.Reshape(letternum_5_shape);
    letternum_5_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    letternum_5_loss_layer_->SetUp(letternum_5_bottom_vec_, letternum_5_top_vec_);
  } else if (letternum_5_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_letter_5_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> letternum_5_shape(1, 1);
    letternum_5_shape.push_back(num_letter_);
    letternum_5_gt_.Reshape(letternum_5_shape);
    letternum_5_pred_.Reshape(letternum_5_shape);
    letternum_5_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    letternum_5_loss_layer_->SetUp(letternum_5_bottom_vec_, letternum_5_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
}

template <typename Dtype>
void MultiBoxccpdLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  num_ = bottom[0]->num();
  num_priors_ = bottom[2]->height() / 4;
  num_gt_ = bottom[10]->height();
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(num_priors_ * loc_classes_ * 4, bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";
  CHECK_EQ(num_priors_ * num_chinese_, bottom[3]->channels())
      << "Number of priors must match number of chinese confidence predictions.";
  CHECK_EQ(num_priors_ * num_english_, bottom[4]->channels())
      << "NUmber of priors must match number of english confidence perdictions.";
  CHECK_EQ(num_priors_ * num_letter_, bottom[5]->channels())
      << "NUmber of priors must match number of letter confidence perdictions.";
}

template <typename Dtype>
void MultiBoxccpdLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* prior_data = bottom[2]->cpu_data();
  const Dtype* chinese_data = bottom[3]->cpu_data();
  const Dtype* english_data = bottom[4]->cpu_data();
  const Dtype* letter_1_data = bottom[5]->cpu_data();
  const Dtype* letter_2_data = bottom[6]->cpu_data();
  const Dtype* letter_3_data = bottom[7]->cpu_data();
  const Dtype* letter_4_data = bottom[8]->cpu_data();
  const Dtype* letter_5_data = bottom[9]->cpu_data();
  const Dtype* gt_data = bottom[10]->cpu_data();

  #if 0
  LOG(INFO)<< "loss compute start printf &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& num_gt_: "<<num_gt_;
  for(int ii=0; ii < num_gt_; ii++)
  {
    int id = ii*10;
    if (gt_data[id] == -1) {
      continue;
    }

    LOG(INFO) <<"LABEL batch_id: "<<gt_data[id]<<" anno_label: "<<gt_data[id+1]
              <<" anno.instance_id: "<<gt_data[id+2];
    LOG(INFO)  <<"LABEL bbox->xmin: "<<gt_data[id+3]<<" bbox->ymin: "<<gt_data[id+4]
              <<" bbox->xmax: "<<gt_data[id+5]<<" bbox->ymax: "<<gt_data[id+6]
              <<" bbox->chinese: "<<gt_data[id+7]<<" bbox->occlusion: "<<gt_data[id+8];
  }
  LOG(INFO)<< "loss compute finished **************************************************** end ";
  
  #endif 

  // Retrieve all ground truth.~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  AnnotatedDatum_AnnoataionAttriType attri_type = AnnotatedDatum_AnnoataionAttriType_LPnumber;
  GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                 &all_gt_bboxes, attri_type);

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  // Retrieve all predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num_, num_priors_, loc_classes_, share_location_,
                    &all_loc_preds);

  // Find matches between source bboxes and ground truth bboxes.
  vector<map<int, vector<float> > > all_match_overlaps;
  FindMatches(all_loc_preds, all_gt_bboxes, prior_bboxes, prior_variances,
              multibox_loss_param_, &all_match_overlaps, &all_match_indices_);

  num_matches_ = 0;
  int num_negs = 0;
  // Sample hard negative (and positive) examples based on mining type.
  MineHardExamples(*bottom[1], all_loc_preds, all_gt_bboxes, prior_bboxes,
                   prior_variances, all_match_overlaps, multibox_loss_param_,
                   &num_matches_, &num_negs, &all_match_indices_,
                   &all_neg_indices_);

  if (num_matches_ >= 1) {
    // Form data to pass on to loc_loss_layer_.
    vector<int> loc_shape(2);
    loc_shape[0] = 1;
    loc_shape[1] = num_matches_ * 4;
    loc_pred_.Reshape(loc_shape);
    loc_gt_.Reshape(loc_shape);
    Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
    Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();
    EncodeLocPrediction(all_loc_preds, all_gt_bboxes, all_match_indices_,
                        prior_bboxes, prior_variances, multibox_loss_param_,
                        loc_pred_data, loc_gt_data);
    loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
    loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);
  } else {
    loc_loss_.mutable_cpu_data()[0] = 0;
  }

  /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // Form data to pass on to conf_loss_layer_.
  if (do_neg_mining_) {
    num_conf_ = num_matches_ + num_negs;
  } else {
    num_conf_ = num_ * num_priors_;
  }
  if (num_conf_ >= 1) {
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~calss confidence loss layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    // Reshape the confidence data.
    vector<int> conf_shape;
    if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      conf_shape.push_back(num_conf_);
      conf_gt_.Reshape(conf_shape);
      conf_shape.push_back(num_classes_);
      conf_pred_.Reshape(conf_shape);
    } else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
      conf_shape.push_back(1);
      conf_shape.push_back(num_conf_);
      conf_shape.push_back(num_classes_);
      conf_gt_.Reshape(conf_shape);
      conf_pred_.Reshape(conf_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    if (!do_neg_mining_) {
      // Consider all scores.
      // Share data and diff with bottom[1].
      CHECK_EQ(conf_pred_.count(), bottom[1]->count());
      conf_pred_.ShareData(*(bottom[1]));
    }
    Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
    Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
    caffe_set(conf_gt_.count(), Dtype(background_label_id_), conf_gt_data);
    EncodeConfPrediction(conf_data, num_, num_priors_, multibox_loss_param_,
                         all_match_indices_, all_neg_indices_, all_gt_bboxes,
                         conf_pred_data, conf_gt_data);
    conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
    conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);

     /*~~~~~~~~~~~~~~~~~~~~~~chinese loss layer  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
     // Reshape the chinese confidence data.
    vector<int> chinesecharcter_shape;
    if (chinesecharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      chinesecharcter_shape.push_back(num_matches_);
      chinesecharcter_gt_.Reshape(chinesecharcter_shape);
      chinesecharcter_shape.push_back(num_chinese_);
      chinesecharcter_pred_.Reshape(chinesecharcter_shape);
    } else if (chinesecharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
      chinesecharcter_shape.push_back(1);
      chinesecharcter_shape.push_back(num_matches_);
      chinesecharcter_shape.push_back(num_chinese_);
      chinesecharcter_gt_.Reshape(chinesecharcter_shape);
      chinesecharcter_pred_.Reshape(chinesecharcter_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    if (!do_neg_mining_) {
      //Share data and diff with bottom[3].
      CHECK_EQ(chinesecharcter_pred_.count(), bottom[3]->count());
      chinesecharcter_pred_.ShareData(*(bottom[3]));
    }
    Dtype* chinesecharcter_pred_data = chinesecharcter_pred_.mutable_cpu_data();
    Dtype* chinesecharcter_gt_data = chinesecharcter_gt_.mutable_cpu_data();
    caffe_set(chinesecharcter_gt_.count(), Dtype(0), chinesecharcter_gt_data);
    EncodeChinConfPrediction(chinese_data, num_, num_priors_, multibox_loss_param_,
                         all_match_indices_, all_neg_indices_, all_gt_bboxes,
                         chinesecharcter_pred_data, chinesecharcter_gt_data);
    chinesecharcter_loss_layer_->Reshape(chinesecharcter_bottom_vec_, chinesecharcter_top_vec_);
    chinesecharcter_loss_layer_->Forward(chinesecharcter_bottom_vec_, chinesecharcter_top_vec_);

    /*~~~~~~~~~~~~~~~~~~~~~~~english_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    // conf english layer
    vector<int> engcharcter_shape;
    if (engcharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      engcharcter_shape.push_back(num_matches_);
      engcharcter_gt_.Reshape(engcharcter_shape);
      engcharcter_shape.push_back(num_english_);
      engcharcter_pred_.Reshape(engcharcter_shape);
    } else if (engcharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
      engcharcter_shape.push_back(1);
      engcharcter_shape.push_back(num_matches_);
      engcharcter_shape.push_back(num_english_);
      engcharcter_gt_.Reshape(engcharcter_shape);
      engcharcter_pred_.Reshape(engcharcter_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    if (!do_neg_mining_) {
      // Consider all scores.
      //t Share daa and diff with bottom[1].
      CHECK_EQ(engcharcter_pred_.count(), bottom[4]->count());
      engcharcter_pred_.ShareData(*(bottom[4]));
    }
    Dtype* engcharcter_pred_data = engcharcter_pred_.mutable_cpu_data();
    Dtype* engcharcter_gt_data = engcharcter_gt_.mutable_cpu_data();
    caffe_set(engcharcter_gt_.count(), Dtype(0), engcharcter_gt_data);
    EncodeEngConfPrediction(english_data, num_, num_priors_, multibox_loss_param_,
                         all_match_indices_, all_neg_indices_, all_gt_bboxes,
                         engcharcter_pred_data, engcharcter_gt_data);
    engcharcter_loss_layer_->Reshape(engcharcter_bottom_vec_, engcharcter_top_vec_);
    engcharcter_loss_layer_->Forward(engcharcter_bottom_vec_, engcharcter_top_vec_);

    /*~~~~~~~~~~~~~~~~~~~~~~~letter_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    // conf letter_1 layer
    vector<int> letter_1_shape;
    if (letternum_1_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      letter_1_shape.push_back(num_matches_);
      letternum_1_gt_.Reshape(letter_1_shape);
      letter_1_shape.push_back(num_letter_);
      letternum_1_pred_.Reshape(letter_1_shape);
    } else if (letternum_1_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
      letter_1_shape.push_back(1);
      letter_1_shape.push_back(num_matches_);
      letter_1_shape.push_back(num_letter_);
      letternum_1_gt_.Reshape(letter_1_shape);
      letternum_1_pred_.Reshape(letter_1_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    if (!do_neg_mining_) {
      // Consider all scores.
      //t Share daa and diff with bottom[1].
      CHECK_EQ(letternum_1_pred_.count(), bottom[5]->count());
      letternum_1_pred_.ShareData(*(bottom[5]));
    }
    Dtype* letternum_1_pred_data = letternum_1_pred_.mutable_cpu_data();
    Dtype* letternum_1_gt_data = letternum_1_gt_.mutable_cpu_data();
    caffe_set(letternum_1_gt_.count(), Dtype(0), letternum_1_gt_data);
    EncodeLettConfPrediction(letter_1_data, num_, num_priors_, multibox_loss_param_,
                         all_match_indices_, all_neg_indices_, all_gt_bboxes,
                         letternum_1_pred_data, letternum_1_gt_data, 1);
    letternum_1_loss_layer_->Reshape(letternum_1_bottom_vec_, letternum_1_top_vec_);
    letternum_1_loss_layer_->Forward(letternum_1_bottom_vec_, letternum_1_top_vec_);

    /*~~~~~~~~~~~~~~~~~~~~~~~letter_2_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    // conf letter_2 layer
    vector<int> letter_2_shape;
    if (letternum_2_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      letter_2_shape.push_back(num_matches_);
      letternum_2_gt_.Reshape(letter_2_shape);
      letter_2_shape.push_back(num_letter_);
      letternum_2_pred_.Reshape(letter_2_shape);
    } else if (letternum_2_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
      letter_2_shape.push_back(1);
      letter_2_shape.push_back(num_matches_);
      letter_2_shape.push_back(num_letter_);
      letternum_2_gt_.Reshape(letter_2_shape);
      letternum_2_pred_.Reshape(letter_2_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    if (!do_neg_mining_) {
      // Consider all scores.
      //t Share daa and diff with bottom[1].
      CHECK_EQ(letternum_2_pred_.count(), bottom[6]->count());
      letternum_2_pred_.ShareData(*(bottom[6]));
    }
    Dtype* letternum_2_pred_data = letternum_2_pred_.mutable_cpu_data();
    Dtype* letternum_2_gt_data = letternum_2_gt_.mutable_cpu_data();
    caffe_set(letternum_2_gt_.count(), Dtype(0), letternum_2_gt_data);
    EncodeLettConfPrediction(letter_2_data, num_, num_priors_, multibox_loss_param_,
                         all_match_indices_, all_neg_indices_, all_gt_bboxes,
                         letternum_2_pred_data, letternum_2_gt_data, 2);
    letternum_2_loss_layer_->Reshape(letternum_2_bottom_vec_, letternum_2_top_vec_);
    letternum_2_loss_layer_->Forward(letternum_2_bottom_vec_, letternum_2_top_vec_);

    /*~~~~~~~~~~~~~~~~~~~~~~~letter_3_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    // conf letter_3 layer
    vector<int> letter_3_shape;
    if (letternum_3_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      letter_3_shape.push_back(num_matches_);
      letternum_3_gt_.Reshape(letter_3_shape);
      letter_3_shape.push_back(num_letter_);
      letternum_3_pred_.Reshape(letter_3_shape);
    } else if (letternum_3_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
      letter_3_shape.push_back(1);
      letter_3_shape.push_back(num_matches_);
      letter_3_shape.push_back(num_letter_);
      letternum_3_gt_.Reshape(letter_3_shape);
      letternum_3_pred_.Reshape(letter_3_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    if (!do_neg_mining_) {
      // Consider all scores.
      //t Share daa and diff with bottom[1].
      CHECK_EQ(letternum_3_pred_.count(), bottom[7]->count());
      letternum_3_pred_.ShareData(*(bottom[7]));
    }
    Dtype* letternum_3_pred_data = letternum_3_pred_.mutable_cpu_data();
    Dtype* letternum_3_gt_data = letternum_3_gt_.mutable_cpu_data();
    caffe_set(letternum_3_gt_.count(), Dtype(0), letternum_3_gt_data);
    EncodeLettConfPrediction(letter_3_data, num_, num_priors_, multibox_loss_param_,
                         all_match_indices_, all_neg_indices_, all_gt_bboxes,
                         letternum_3_pred_data, letternum_3_gt_data, 3);
    letternum_3_loss_layer_->Reshape(letternum_3_bottom_vec_, letternum_3_top_vec_);
    letternum_3_loss_layer_->Forward(letternum_3_bottom_vec_, letternum_3_top_vec_);

    /*~~~~~~~~~~~~~~~~~~~~~~~letter_4_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    // conf letter_4 layer
    vector<int> letter_4_shape;
    if (letternum_4_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      letter_4_shape.push_back(num_matches_);
      letternum_4_gt_.Reshape(letter_4_shape);
      letter_4_shape.push_back(num_letter_);
      letternum_4_pred_.Reshape(letter_4_shape);
    } else if (letternum_4_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
      letter_4_shape.push_back(1);
      letter_4_shape.push_back(num_matches_);
      letter_4_shape.push_back(num_letter_);
      letternum_4_gt_.Reshape(letter_4_shape);
      letternum_4_pred_.Reshape(letter_4_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    if (!do_neg_mining_) {
      // Consider all scores.
      //t Share daa and diff with bottom[1].
      CHECK_EQ(letternum_4_pred_.count(), bottom[8]->count());
      letternum_4_pred_.ShareData(*(bottom[8]));
    }
    Dtype* letternum_4_pred_data = letternum_4_pred_.mutable_cpu_data();
    Dtype* letternum_4_gt_data = letternum_4_gt_.mutable_cpu_data();
    caffe_set(letternum_4_gt_.count(), Dtype(0), letternum_4_gt_data);
    EncodeLettConfPrediction(letter_4_data, num_, num_priors_, multibox_loss_param_,
                         all_match_indices_, all_neg_indices_, all_gt_bboxes,
                         letternum_4_pred_data, letternum_4_gt_data, 4);
    letternum_4_loss_layer_->Reshape(letternum_4_bottom_vec_, letternum_4_top_vec_);
    letternum_4_loss_layer_->Forward(letternum_4_bottom_vec_, letternum_4_top_vec_);

    /*~~~~~~~~~~~~~~~~~~~~~~~letter_5_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    // conf letter_5 layer
    vector<int> letter_5_shape;
    if (letternum_5_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      letter_5_shape.push_back(num_matches_);
      letternum_5_gt_.Reshape(letter_5_shape);
      letter_5_shape.push_back(num_letter_);
      letternum_5_pred_.Reshape(letter_5_shape);
    } else if (letternum_5_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
      letter_5_shape.push_back(1);
      letter_5_shape.push_back(num_matches_);
      letter_5_shape.push_back(num_letter_);
      letternum_5_gt_.Reshape(letter_5_shape);
      letternum_5_pred_.Reshape(letter_5_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    if (!do_neg_mining_) {
      // Consider all scores.
      //t Share daa and diff with bottom[1].
      CHECK_EQ(letternum_5_pred_.count(), bottom[9]->count());
      letternum_5_pred_.ShareData(*(bottom[9]));
    }
    Dtype* letternum_5_pred_data = letternum_5_pred_.mutable_cpu_data();
    Dtype* letternum_5_gt_data = letternum_5_gt_.mutable_cpu_data();
    caffe_set(letternum_5_gt_.count(), Dtype(0), letternum_5_gt_data);
    EncodeLettConfPrediction(letter_5_data, num_, num_priors_, multibox_loss_param_,
                         all_match_indices_, all_neg_indices_, all_gt_bboxes,
                         letternum_5_pred_data, letternum_5_gt_data, 5);
    letternum_5_loss_layer_->Reshape(letternum_5_bottom_vec_, letternum_5_top_vec_);
    letternum_5_loss_layer_->Forward(letternum_5_bottom_vec_, letternum_5_top_vec_);
  } else {
    conf_loss_.mutable_cpu_data()[0] = 0;
    chinesecharcter_loss_.mutable_cpu_data()[0] = 0;
    engcharcter_loss_.mutable_cpu_data()[0] = 0;
    letternum_1_loss_.mutable_cpu_data()[0] = 0;
    letternum_2_loss_.mutable_cpu_data()[0] = 0;
    letternum_3_loss_.mutable_cpu_data()[0] = 0;
    letternum_4_loss_.mutable_cpu_data()[0] = 0;
    letternum_5_loss_.mutable_cpu_data()[0] = 0;
  }
  /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  top[0]->mutable_cpu_data()[0] = 0;
  Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_priors_, num_matches_);
  if (this->layer_param_.propagate_down(0)) {
    top[0]->mutable_cpu_data()[0] +=
        loc_weight_ * loc_loss_.cpu_data()[0] / normalizer;
  }
  if (this->layer_param_.propagate_down(1)) {
    top[0]->mutable_cpu_data()[0] += 
          conf_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(3)) {
    top[0]->mutable_cpu_data()[0] += 
          0.5*chinesecharcter_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(4)) {
    top[0]->mutable_cpu_data()[0] += 
          0.5*engcharcter_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(5)) {
    top[0]->mutable_cpu_data()[0] += 
          0.5*letternum_1_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(6)) {
    top[0]->mutable_cpu_data()[0] += 
          0.5*letternum_2_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(7)) {
    top[0]->mutable_cpu_data()[0] += 
          0.5*letternum_3_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(8)) {
    top[0]->mutable_cpu_data()[0] += 
          0.5*letternum_4_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(9)) {
    top[0]->mutable_cpu_data()[0] += 
          0.5*letternum_5_loss_.cpu_data()[0] / normalizer;
  }
  #if 0
  LOG(INFO)<<"num_matches_: "<<num_matches_<<" num_gt_boxes: "<<num_gt_<<" num_conf_: "<<num_conf_;
  LOG(INFO)<<"origin loc_loss_: "<< loc_loss_.cpu_data()[0];
  LOG(INFO)<<"origin conf_loss_: "<<conf_loss_.cpu_data()[0];
  LOG(INFO)<<"origin chinesecharector_loss_: "<<chinesecharcter_loss_.cpu_data()[0];
  LOG(INFO)<<"origin engcharcter_loss_: " <<engcharcter_loss_.cpu_data()[0];
  LOG(INFO)<<"total ~~~~~~~~~~~~~~~~~~loss: "<<top[0]->mutable_cpu_data()[0]<<" normalizer: "<<normalizer;
  #endif
}

template <typename Dtype>
void MultiBoxccpdLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to prior inputs.";
  }
  if (propagate_down[10]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  // Back propagate on location prediction.
  if (propagate_down[0]) {
    Dtype* loc_bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), loc_bottom_diff);
    if (num_matches_ >= 1) {
      vector<bool> loc_propagate_down;
      // Only back propagate on prediction, not ground truth.
      loc_propagate_down.push_back(true);
      loc_propagate_down.push_back(false);
      loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down,
                                loc_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(loc_pred_.count(), loss_weight, loc_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[0].
      const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
      int count = 0;
      for (int i = 0; i < num_; ++i) {
        for (map<int, vector<int> >::iterator it =
             all_match_indices_[i].begin();
             it != all_match_indices_[i].end(); ++it) {
          const int label = share_location_ ? 0 : it->first;
          const vector<int>& match_index = it->second;
          for (int j = 0; j < match_index.size(); ++j) {
            if (match_index[j] <= -1) {
              continue;
            }
            // Copy the diff to the right place.
            int start_idx = loc_classes_ * 4 * j + label * 4;
            caffe_copy<Dtype>(4, loc_pred_diff + count * 4,
                              loc_bottom_diff + start_idx);
            ++count;
          }
        }
        loc_bottom_diff += bottom[0]->offset(1);
      }
    }
  }

  // Back propagate on confidence prediction.
  if (propagate_down[1]) {
    Dtype* conf_bottom_diff = bottom[1]->mutable_cpu_diff();
    caffe_set(bottom[1]->count(), Dtype(0), conf_bottom_diff);
    if (num_conf_ >= 1) {
      vector<bool> conf_propagate_down;
      // Only back propagate on prediction, not ground truth.
      conf_propagate_down.push_back(true);
      conf_propagate_down.push_back(false);
      conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down,
                                 conf_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(conf_pred_.count(), loss_weight,
                 conf_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[1].
      const Dtype* conf_pred_diff = conf_pred_.cpu_diff();
      if (do_neg_mining_) {
        int count = 0;
        for (int i = 0; i < num_; ++i) {
          // Copy matched (positive) bboxes scores' diff.
          const map<int, vector<int> >& match_indices = all_match_indices_[i];
          for (map<int, vector<int> >::const_iterator it =
               match_indices.begin(); it != match_indices.end(); ++it) {
            const vector<int>& match_index = it->second;
            CHECK_EQ(match_index.size(), num_priors_);
            for (int j = 0; j < num_priors_; ++j) {
              if (match_index[j] <= -1) {
                continue;
              }
              // Copy the diff to the right place.
              caffe_copy<Dtype>(num_classes_,
                                conf_pred_diff + count * num_classes_,
                                conf_bottom_diff + j * num_classes_);
              ++count;
            }
          }
          // Copy negative bboxes scores' diff.
          for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
            int j = all_neg_indices_[i][n];
            CHECK_LT(j, num_priors_);
            caffe_copy<Dtype>(num_classes_,
                              conf_pred_diff + count * num_classes_,
                              conf_bottom_diff + j * num_classes_);
            ++count;
          }
          conf_bottom_diff += bottom[1]->offset(1);
        }
      } else {
        bottom[1]->ShareDiff(conf_pred_);
      }
    }
  }

  // Back propagate on chinesecharcter_loss_ prediction.
  if (propagate_down[3]) {
    Dtype* chinese_bottom_diff = bottom[3]->mutable_cpu_diff();
    caffe_set(bottom[3]->count(), Dtype(0), chinese_bottom_diff);
    if (num_conf_ >= 1) {
      vector<bool> chinese_propagate_down;
      // Only back propagate on prediction, not ground truth.
      chinese_propagate_down.push_back(true);
      chinese_propagate_down.push_back(false);
      chinesecharcter_loss_layer_->Backward(chinesecharcter_top_vec_, chinese_propagate_down,
                                 chinesecharcter_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(chinesecharcter_pred_.count(), loss_weight,
                 chinesecharcter_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[1].
      const Dtype* chinesecharcter_pred_diff = chinesecharcter_pred_.cpu_diff();
      if (do_neg_mining_) {
        int count = 0;
        for (int i = 0; i < num_; ++i) {
          // Copy matched (positive) bboxes scores' diff.
          const map<int, vector<int> >& match_indices = all_match_indices_[i];
          for (map<int, vector<int> >::const_iterator it =
               match_indices.begin(); it != match_indices.end(); ++it) {
            const vector<int>& match_index = it->second;
            CHECK_EQ(match_index.size(), num_priors_);
            for (int j = 0; j < num_priors_; ++j) {
              if (match_index[j] <= -1) {
                continue;
              }
              // Copy the diff to the right place.
              caffe_copy<Dtype>(num_chinese_,
                                chinesecharcter_pred_diff + count * num_chinese_,
                                chinese_bottom_diff + j * num_chinese_);
              ++count;
            }
          }
          chinese_bottom_diff += bottom[3]->offset(1);
        }
      } else {
        // The diff is already computed and stored.
        bottom[3]->ShareDiff(chinesecharcter_pred_);
      }
    }
  }

  // Back propagate on english prediction.
  if (propagate_down[4]) {
    Dtype* engcharcter_bottom_diff = bottom[4]->mutable_cpu_diff();
    caffe_set(bottom[4]->count(), Dtype(0), engcharcter_bottom_diff);
    if (num_conf_ >= 1) {
      vector<bool> english_propagate_down;
      // Only back propagate on prediction, not ground truth.
      english_propagate_down.push_back(true);
      english_propagate_down.push_back(false);
      engcharcter_loss_layer_->Backward(engcharcter_top_vec_, english_propagate_down,
                                 engcharcter_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(engcharcter_pred_.count(), loss_weight,
                 engcharcter_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[4].
      const Dtype* engcharcter_pred_diff = engcharcter_pred_.cpu_diff();
      if (do_neg_mining_) {
        int count = 0;
        for (int i = 0; i < num_; ++i) {
          // Copy matched (positive) bboxes scores' diff.
          const map<int, vector<int> >& match_indices = all_match_indices_[i];
          for (map<int, vector<int> >::const_iterator it =
               match_indices.begin(); it != match_indices.end(); ++it) {
            const vector<int>& match_index = it->second;
            CHECK_EQ(match_index.size(), num_priors_);
            for (int j = 0; j < num_priors_; ++j) {
              if (match_index[j] <= -1) {
                continue;
              }
              // Copy the diff to the right place.
              caffe_copy<Dtype>(num_english_,
                                engcharcter_pred_diff + count * num_english_,
                                engcharcter_bottom_diff + j * num_english_);
              ++count;
            }
          }
          engcharcter_bottom_diff += bottom[4]->offset(1);
        }
      } else {
        // The diff is already computed and stored.
        bottom[4]->ShareDiff(engcharcter_pred_);
      }
    }
  }

  // Back propagate on letter prediction.
  if (propagate_down[5]) {
    Dtype* letter_1_bottom_diff = bottom[5]->mutable_cpu_diff();
    caffe_set(bottom[5]->count(), Dtype(0), letter_1_bottom_diff);
    if (num_conf_ >= 1) {
      vector<bool> letter_1_propagate_down;
      // Only back propagate on prediction, not ground truth.
      letter_1_propagate_down.push_back(true);
      letter_1_propagate_down.push_back(false);
      letternum_1_loss_layer_->Backward(letternum_1_top_vec_, letter_1_propagate_down,
                                 letternum_1_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(letternum_1_pred_.count(), loss_weight,
                 letternum_1_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[4].
      const Dtype* letternum_1_pred_diff = letternum_1_pred_.cpu_diff();
      if (do_neg_mining_) {
        int count = 0;
        for (int i = 0; i < num_; ++i) {
          // Copy matched (positive) bboxes scores' diff.
          const map<int, vector<int> >& match_indices = all_match_indices_[i];
          for (map<int, vector<int> >::const_iterator it =
               match_indices.begin(); it != match_indices.end(); ++it) {
            const vector<int>& match_index = it->second;
            CHECK_EQ(match_index.size(), num_priors_);
            for (int j = 0; j < num_priors_; ++j) {
              if (match_index[j] <= -1) {
                continue;
              }
              // Copy the diff to the right place.
              caffe_copy<Dtype>(num_letter_,
                                letternum_1_pred_diff + count * num_letter_,
                                letter_1_bottom_diff + j * num_letter_);
              ++count;
            }
          }
          letter_1_bottom_diff += bottom[5]->offset(1);
        }
      } else {
        // The diff is already computed and stored.
        bottom[5]->ShareDiff(letternum_1_pred_);
      }
    }
  }

  // Back propagate on letter 2 prediction.
  if (propagate_down[6]) {
    Dtype* letter_2_bottom_diff = bottom[6]->mutable_cpu_diff();
    caffe_set(bottom[6]->count(), Dtype(0), letter_2_bottom_diff);
    if (num_conf_ >= 1) {
      vector<bool> letter_2_propagate_down;
      // Only back propagate on prediction, not ground truth.
      letter_2_propagate_down.push_back(true);
      letter_2_propagate_down.push_back(false);
      letternum_2_loss_layer_->Backward(letternum_2_top_vec_, letter_2_propagate_down,
                                 letternum_2_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(letternum_2_pred_.count(), loss_weight,
                 letternum_2_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[4].
      const Dtype* letternum_2_pred_diff = letternum_2_pred_.cpu_diff();
      if (do_neg_mining_) {
        int count = 0;
        for (int i = 0; i < num_; ++i) {
          // Copy matched (positive) bboxes scores' diff.
          const map<int, vector<int> >& match_indices = all_match_indices_[i];
          for (map<int, vector<int> >::const_iterator it =
               match_indices.begin(); it != match_indices.end(); ++it) {
            const vector<int>& match_index = it->second;
            CHECK_EQ(match_index.size(), num_priors_);
            for (int j = 0; j < num_priors_; ++j) {
              if (match_index[j] <= -1) {
                continue;
              }
              // Copy the diff to the right place.
              caffe_copy<Dtype>(num_letter_,
                                letternum_2_pred_diff + count * num_letter_,
                                letter_2_bottom_diff + j * num_letter_);
              ++count;
            }
          }
          letter_2_bottom_diff += bottom[6]->offset(1);
        }
      } else {
        // The diff is already computed and stored.
        bottom[6]->ShareDiff(letternum_2_pred_);
      }
    }
  }

// Back propagate on letter 3 prediction.
  if (propagate_down[7]) {
    Dtype* letter_3_bottom_diff = bottom[7]->mutable_cpu_diff();
    caffe_set(bottom[7]->count(), Dtype(0), letter_3_bottom_diff);
    if (num_conf_ >= 1) {
      vector<bool> letter_3_propagate_down;
      // Only back propagate on prediction, not ground truth.
      letter_3_propagate_down.push_back(true);
      letter_3_propagate_down.push_back(false);
      letternum_3_loss_layer_->Backward(letternum_3_top_vec_, letter_3_propagate_down,
                                 letternum_3_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(letternum_3_pred_.count(), loss_weight,
                 letternum_3_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[4].
      const Dtype* letternum_3_pred_diff = letternum_3_pred_.cpu_diff();
      if (do_neg_mining_) {
        int count = 0;
        for (int i = 0; i < num_; ++i) {
          // Copy matched (positive) bboxes scores' diff.
          const map<int, vector<int> >& match_indices = all_match_indices_[i];
          for (map<int, vector<int> >::const_iterator it =
               match_indices.begin(); it != match_indices.end(); ++it) {
            const vector<int>& match_index = it->second;
            CHECK_EQ(match_index.size(), num_priors_);
            for (int j = 0; j < num_priors_; ++j) {
              if (match_index[j] <= -1) {
                continue;
              }
              // Copy the diff to the right place.
              caffe_copy<Dtype>(num_letter_,
                                letternum_3_pred_diff + count * num_letter_,
                                letter_3_bottom_diff + j * num_letter_);
              ++count;
            }
          }
          letter_3_bottom_diff += bottom[7]->offset(1);
        }
      } else {
        // The diff is already computed and stored.
        bottom[7]->ShareDiff(letternum_3_pred_);
      }
    }
  }

  // Back propagate on letter 4 prediction.
  if (propagate_down[8]) {
    Dtype* letter_4_bottom_diff = bottom[8]->mutable_cpu_diff();
    caffe_set(bottom[8]->count(), Dtype(0), letter_4_bottom_diff);
    if (num_conf_ >= 1) {
      vector<bool> letter_4_propagate_down;
      // Only back propagate on prediction, not ground truth.
      letter_4_propagate_down.push_back(true);
      letter_4_propagate_down.push_back(false);
      letternum_4_loss_layer_->Backward(letternum_4_top_vec_, letter_4_propagate_down,
                                 letternum_4_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(letternum_4_pred_.count(), loss_weight,
                 letternum_4_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[4].
      const Dtype* letternum_4_pred_diff = letternum_4_pred_.cpu_diff();
      if (do_neg_mining_) {
        int count = 0;
        for (int i = 0; i < num_; ++i) {
          // Copy matched (positive) bboxes scores' diff.
          const map<int, vector<int> >& match_indices = all_match_indices_[i];
          for (map<int, vector<int> >::const_iterator it =
               match_indices.begin(); it != match_indices.end(); ++it) {
            const vector<int>& match_index = it->second;
            CHECK_EQ(match_index.size(), num_priors_);
            for (int j = 0; j < num_priors_; ++j) {
              if (match_index[j] <= -1) {
                continue;
              }
              // Copy the diff to the right place.
              caffe_copy<Dtype>(num_letter_,
                                letternum_4_pred_diff + count * num_letter_,
                                letter_4_bottom_diff + j * num_letter_);
              ++count;
            }
          }
          letter_4_bottom_diff += bottom[8]->offset(1);
        }
      } else {
        // The diff is already computed and stored.
        bottom[8]->ShareDiff(letternum_4_pred_);
      }
    }
  }

  // Back propagate on letter 5 prediction.
  if (propagate_down[9]) {
    Dtype* letter_5_bottom_diff = bottom[9]->mutable_cpu_diff();
    caffe_set(bottom[9]->count(), Dtype(0), letter_5_bottom_diff);
    if (num_conf_ >= 1) {
      vector<bool> letter_5_propagate_down;
      // Only back propagate on prediction, not ground truth.
      letter_5_propagate_down.push_back(true);
      letter_5_propagate_down.push_back(false);
      letternum_5_loss_layer_->Backward(letternum_5_top_vec_, letter_5_propagate_down,
                                 letternum_5_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(letternum_5_pred_.count(), loss_weight,
                 letternum_5_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[4].
      const Dtype* letternum_5_pred_diff = letternum_5_pred_.cpu_diff();
      if (do_neg_mining_) {
        int count = 0;
        for (int i = 0; i < num_; ++i) {
          // Copy matched (positive) bboxes scores' diff.
          const map<int, vector<int> >& match_indices = all_match_indices_[i];
          for (map<int, vector<int> >::const_iterator it =
               match_indices.begin(); it != match_indices.end(); ++it) {
            const vector<int>& match_index = it->second;
            CHECK_EQ(match_index.size(), num_priors_);
            for (int j = 0; j < num_priors_; ++j) {
              if (match_index[j] <= -1) {
                continue;
              }
              // Copy the diff to the right place.
              caffe_copy<Dtype>(num_letter_,
                                letternum_5_pred_diff + count * num_letter_,
                                letter_5_bottom_diff + j * num_letter_);
              ++count;
            }
          }
          letter_5_bottom_diff += bottom[9]->offset(1);
        }
      } else {
        // The diff is already computed and stored.
        bottom[9]->ShareDiff(letternum_5_pred_);
      }
    }
  }
  // After backward, remove match statistics.
  all_match_indices_.clear();
  all_neg_indices_.clear();
}

INSTANTIATE_CLASS(MultiBoxccpdLossLayer);
REGISTER_LAYER_CLASS(MultiBoxccpdLoss);

}  // namespace caffe
