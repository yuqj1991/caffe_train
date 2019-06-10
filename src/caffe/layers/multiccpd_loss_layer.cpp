#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multiccpd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MulticcpdLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
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
  batch_size_ = bottom[0]->num();
  // Get other parameters.
  CHECK(multibox_loss_param.has_num_chinese()) << "Must prodived num_chinese";
  CHECK(multibox_loss_param.has_num_english()) << "Must provide num_english";
  num_chinese_ = multibox_loss_param.num_chinese();
  num_english_ = multibox_loss_param.num_english();
  num_letter_ = multibox_loss_param.num_letter();

  if (!this->layer_param_.loss_param().has_normalization() &&
    this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                    LossParameter_NormalizationMode_VALID :
                    LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  vector<int> loss_shape(1, 1);
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
void MulticcpdLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  int num_chin = bottom[0]->shape(1);  //bottom[0]: landmarks , 
  int num_eng = bottom[1]->shape(1); //bottom[1]: num_gender;
  int num_letter = bottom[2]->shape(1); // bottom[2]: num_glasses;
  CHECK_EQ(num_chin, num_chinese_)<<"number of num_chinese_ point value must equal";
  CHECK_EQ(num_eng, num_english_)<<"number of num_english_ must match prototxt provided";
  CHECK_EQ(num_letter, num_letter_)<<"number of num_letter_ must match prototxt provided";
}

template <typename Dtype>
void MulticcpdLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* gt_data = bottom[7]->cpu_data();
  /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  vector<int> all_chi;
  vector<int> all_eng;
  vector<int> all_let_1;
  vector<int> all_let_2;
  vector<int> all_let_3;
  vector<int> all_let_4;
  vector<int> all_let_5;
  for(int item_id =0; item_id<batch_size_; item_id++){
    int idxg = item_id*7;
    all_chi.push_back(gt_data[idxg+0]);
    all_eng.push_back(gt_data[idxg+1]);
    all_let_1.push_back(gt_data[idxg+2]);
    all_let_2.push_back(gt_data[idxg+3]);
    all_let_3.push_back(gt_data[idxg+4]);
    all_let_4.push_back(gt_data[idxg+5]);
    all_let_5.push_back(gt_data[idxg+6]);
  }
  #if 0
  LOG(INFO)<<"==================loss layer groundth======";
  LOG(INFO)<<"batch_size_: "<<batch_size_;
  for(int idx =0; idx<batch_size_; idx++){
    LOG(INFO) <<" chi: "<<all_chi[idx]
              <<" eng: "<<all_eng[idx]<<" let1: "<<all_let_1[idx]<<" let2: "<<all_let_2[idx]
              <<" let3: "<<all_let_3[idx]<<" let4: "<<all_let_4[idx]
              <<" let5: "<<all_let_5[idx];
  }
  LOG(INFO)<<"=================END=======================";
  #endif
  
  /*~~~~~~~~~~~~~~~~~~~~~~chinese loss layer  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // Reshape the chinese confidence data.
  vector<int> chinesecharcter_shape;
  if (chinesecharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    chinesecharcter_shape.push_back(batch_size_);
    chinesecharcter_gt_.Reshape(chinesecharcter_shape);
    chinesecharcter_shape.push_back(num_chinese_);
    chinesecharcter_pred_.Reshape(chinesecharcter_shape);
    //chinesecharcter_pred_.CopyFrom(*bottom[0]);
  } else if (chinesecharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    //chinesecharcter_shape.push_back(1);
    chinesecharcter_shape.push_back(batch_size_);
    chinesecharcter_shape.push_back(num_chinese_);
    chinesecharcter_gt_.Reshape(chinesecharcter_shape);
    chinesecharcter_pred_.Reshape(chinesecharcter_shape);
    /************************************************/
    //chinesecharcter_pred_.CopyFrom(*bottom[0]);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  Dtype* chinesecharcter_gt_data = chinesecharcter_gt_.mutable_cpu_data();
  caffe_set(chinesecharcter_gt_.count(), Dtype(0), chinesecharcter_gt_data);
  for(int ii=0;ii<batch_size_; ii++){
    switch (chinesecharcter_loss_type_)
    {
    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
      chinesecharcter_gt_data[ii] = all_chi[ii];
      break;
    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
      chinesecharcter_gt_data[ii * num_chinese_ + all_chi[ii]] = 1;
      break;
    default:
      LOG(FATAL) << "Unknown conf loss type.";
    }
  }
  Dtype* chinesecharcter_pred_data = chinesecharcter_pred_.mutable_cpu_data();
  const Dtype* chinese_data = bottom[0]->cpu_data();
  caffe_copy<Dtype>(bottom[0]->count(), chinese_data, chinesecharcter_pred_data);
  chinesecharcter_loss_layer_->Reshape(chinesecharcter_bottom_vec_, chinesecharcter_top_vec_);
  chinesecharcter_loss_layer_->Forward(chinesecharcter_bottom_vec_, chinesecharcter_top_vec_);

  /*~~~~~~~~~~~~~~~~~~~~~~~english_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // conf english layer
  vector<int> engcharcter_shape;
  if (engcharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    engcharcter_shape.push_back(batch_size_);
    engcharcter_gt_.Reshape(engcharcter_shape);
    engcharcter_shape.push_back(num_english_);
    engcharcter_pred_.Reshape(engcharcter_shape);
    //engcharcter_pred_.CopyFrom(*bottom[1]);
  } else if (engcharcter_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    //engcharcter_shape.push_back(1);
    engcharcter_shape.push_back(batch_size_);
    engcharcter_shape.push_back(num_english_);
    engcharcter_gt_.Reshape(engcharcter_shape);
    engcharcter_pred_.Reshape(engcharcter_shape);
    /************************************************/
    //engcharcter_pred_.CopyFrom(*bottom[1]);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  Dtype* engcharcter_gt_data = engcharcter_gt_.mutable_cpu_data();
  caffe_set(engcharcter_gt_.count(), Dtype(0), engcharcter_gt_data);
  for(int ii=0;ii<batch_size_; ii++){
    switch (engcharcter_loss_type_)
    {
    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
      engcharcter_gt_data[ii] = all_eng[ii];
      break;
    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
      engcharcter_gt_data[ii * num_english_ + all_eng[ii]] = 1;
      break;
    default:
      LOG(FATAL) << "Unknown conf loss type.";
    }
  }
  Dtype* engcharcter_pred_data = engcharcter_pred_.mutable_cpu_data();
  const Dtype* eng_data = bottom[1]->cpu_data();
  caffe_copy<Dtype>(bottom[1]->count(), eng_data, engcharcter_pred_data);
  engcharcter_loss_layer_->Reshape(engcharcter_bottom_vec_, engcharcter_top_vec_);
  engcharcter_loss_layer_->Forward(engcharcter_bottom_vec_, engcharcter_top_vec_);

  /*~~~~~~~~~~~~~~~~~~~~~~~letter_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // conf letter_1 layer
  vector<int> letter_1_shape;
  if (letternum_1_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    letter_1_shape.push_back(batch_size_);
    letternum_1_gt_.Reshape(letter_1_shape);
    letter_1_shape.push_back(num_letter_);
    letternum_1_pred_.Reshape(letter_1_shape);
    //letternum_1_pred_.CopyFrom(*bottom[2]);
  } else if (letternum_1_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    //letter_1_shape.push_back(1);
    letter_1_shape.push_back(batch_size_);
    letter_1_shape.push_back(num_letter_);
    letternum_1_gt_.Reshape(letter_1_shape);
    letternum_1_pred_.Reshape(letter_1_shape);
    /************************************************/
    //letternum_1_pred_.CopyFrom(*bottom[2]);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  Dtype* letternum_1_gt_data = letternum_1_gt_.mutable_cpu_data();
  caffe_set(letternum_1_gt_.count(), Dtype(0), letternum_1_gt_data);
  for(int ii=0;ii<batch_size_; ii++){
    switch (letternum_1_loss_type_)
    {
    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
      letternum_1_gt_data[ii] = all_let_1[ii];
      break;
    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
      letternum_1_gt_data[ii * num_letter_ + all_let_1[ii]] = 1;
      break;
    default:
      LOG(FATAL) << "Unknown conf loss type.";
    }
  }
  Dtype* letternum_1_pred_data = letternum_1_pred_.mutable_cpu_data();
  const Dtype* let_1_data = bottom[2]->cpu_data();
  caffe_copy<Dtype>(bottom[2]->count(), let_1_data, letternum_1_pred_data);
  letternum_1_loss_layer_->Reshape(letternum_1_bottom_vec_, letternum_1_top_vec_);
  letternum_1_loss_layer_->Forward(letternum_1_bottom_vec_, letternum_1_top_vec_);

  /*~~~~~~~~~~~~~~~~~~~~~~~letter_2_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // conf letter_2 layer
  vector<int> letter_2_shape;
  if (letternum_2_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    letter_2_shape.push_back(batch_size_);
    letternum_2_gt_.Reshape(letter_2_shape);
    letter_2_shape.push_back(num_letter_);
    letternum_2_pred_.Reshape(letter_2_shape);
    //letternum_2_pred_.CopyFrom(*bottom[3]);
  } else if (letternum_2_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    //letter_2_shape.push_back(1);
    letter_2_shape.push_back(batch_size_);
    letter_2_shape.push_back(num_letter_);
    letternum_2_gt_.Reshape(letter_2_shape);
    letternum_2_pred_.Reshape(letter_2_shape);
    /************************************************/
    //letternum_2_pred_.CopyFrom(*bottom[3]);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  Dtype* letternum_2_gt_data = letternum_2_gt_.mutable_cpu_data();
  caffe_set(letternum_2_gt_.count(), Dtype(0), letternum_2_gt_data);
  for(int ii=0;ii<batch_size_; ii++){
    switch (letternum_2_loss_type_)
    {
    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
      letternum_2_gt_data[ii] = all_let_2[ii];
      break;
    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
      letternum_2_gt_data[ii * num_letter_ + all_let_2[ii]] = 1;
      break;
    default:
      LOG(FATAL) << "Unknown conf loss type.";
    }
  }
  Dtype* letternum_2_pred_data = letternum_2_pred_.mutable_cpu_data();
  const Dtype* let_2_data = bottom[3]->cpu_data();
  caffe_copy<Dtype>(bottom[3]->count(), let_2_data, letternum_2_pred_data);
  letternum_2_loss_layer_->Reshape(letternum_2_bottom_vec_, letternum_2_top_vec_);
  letternum_2_loss_layer_->Forward(letternum_2_bottom_vec_, letternum_2_top_vec_);

  /*~~~~~~~~~~~~~~~~~~~~~~~letter_3_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // conf letter_3 layer
  vector<int> letter_3_shape;
  if (letternum_3_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    letter_3_shape.push_back(batch_size_);
    letternum_3_gt_.Reshape(letter_3_shape);
    letter_3_shape.push_back(num_letter_);
    letternum_3_pred_.Reshape(letter_3_shape);
    //letternum_3_pred_.CopyFrom(*bottom[4]);
  } else if (letternum_3_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    //letter_3_shape.push_back(1);
    letter_3_shape.push_back(batch_size_);
    letter_3_shape.push_back(num_letter_);
    letternum_3_gt_.Reshape(letter_3_shape);
    letternum_3_pred_.Reshape(letter_3_shape);
    /************************************************/
    //letternum_3_pred_.CopyFrom(*bottom[4]);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  Dtype* letternum_3_gt_data = letternum_3_gt_.mutable_cpu_data();
  caffe_set(letternum_3_gt_.count(), Dtype(0), letternum_3_gt_data);
  for(int ii=0;ii<batch_size_; ii++){
    switch (letternum_3_loss_type_)
    {
    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
      letternum_3_gt_data[ii] = all_let_3[ii];
      break;
    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
      letternum_3_gt_data[ii * num_letter_ + all_let_3[ii]] = 1;
      break;
    default:
      LOG(FATAL) << "Unknown conf loss type.";
    }
  }
  Dtype* letternum_3_pred_data = letternum_3_pred_.mutable_cpu_data();
  const Dtype* let_3_data = bottom[4]->cpu_data();
  caffe_copy<Dtype>(bottom[4]->count(), let_3_data, letternum_3_pred_data);
  letternum_3_loss_layer_->Reshape(letternum_3_bottom_vec_, letternum_3_top_vec_);
  letternum_3_loss_layer_->Forward(letternum_3_bottom_vec_, letternum_3_top_vec_);

  /*~~~~~~~~~~~~~~~~~~~~~~~letter_4_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // conf letter_4 layer
  vector<int> letter_4_shape;
  if (letternum_4_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    letter_4_shape.push_back(batch_size_);
    letternum_4_gt_.Reshape(letter_4_shape);
    letter_4_shape.push_back(num_letter_);
    letternum_4_pred_.Reshape(letter_4_shape);
    //letternum_4_pred_.CopyFrom(*bottom[5]);
  } else if (letternum_4_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    //letter_4_shape.push_back(1);
    letter_4_shape.push_back(batch_size_);
    letter_4_shape.push_back(num_letter_);
    letternum_4_gt_.Reshape(letter_4_shape);
    letternum_4_pred_.Reshape(letter_4_shape);
    /************************************************/
    //letternum_4_pred_.CopyFrom(*bottom[5]);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  Dtype* letternum_4_gt_data = letternum_4_gt_.mutable_cpu_data();
  caffe_set(letternum_4_gt_.count(), Dtype(0), letternum_4_gt_data);
  for(int ii=0;ii<batch_size_; ii++){
    switch (letternum_4_loss_type_)
    {
    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
      letternum_4_gt_data[ii] = all_let_4[ii];
      break;
    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
      letternum_4_gt_data[ii * num_letter_ + all_let_4[ii]] = 1;
      break;
    default:
      LOG(FATAL) << "Unknown conf loss type.";
    }
  }
  Dtype* letternum_4_pred_data = letternum_4_pred_.mutable_cpu_data();
  const Dtype* let_4_data = bottom[5]->cpu_data();
  caffe_copy<Dtype>(bottom[5]->count(), let_4_data, letternum_4_pred_data);
  letternum_4_loss_layer_->Reshape(letternum_4_bottom_vec_, letternum_4_top_vec_);
  letternum_4_loss_layer_->Forward(letternum_4_bottom_vec_, letternum_4_top_vec_);

  /*~~~~~~~~~~~~~~~~~~~~~~~letter_5_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // conf letter_5 layer
  vector<int> letter_5_shape;
  if (letternum_5_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    letter_5_shape.push_back(batch_size_);
    letternum_5_gt_.Reshape(letter_5_shape);
    letter_5_shape.push_back(num_letter_);
    letternum_5_pred_.Reshape(letter_5_shape);
    //letternum_5_pred_.CopyFrom(*bottom[6]);
  } else if (letternum_5_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    //letter_5_shape.push_back(1);
    letter_5_shape.push_back(batch_size_);
    letter_5_shape.push_back(num_letter_);
    letternum_5_gt_.Reshape(letter_5_shape);
    letternum_5_pred_.Reshape(letter_5_shape);
    /************************************************/
    //letternum_5_pred_.CopyFrom(*bottom[6]);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  Dtype* letternum_5_gt_data = letternum_5_gt_.mutable_cpu_data();
  caffe_set(letternum_5_gt_.count(), Dtype(0), letternum_5_gt_data);
  for(int ii=0;ii<batch_size_; ii++){
    switch (letternum_5_loss_type_)
    {
    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
      letternum_5_gt_data[ii] = all_let_5[ii];
      break;
    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
      letternum_5_gt_data[ii * num_letter_ + all_let_5[ii]] = 1;
      break;
    default:
      LOG(FATAL) << "Unknown conf loss type.";
    }
  }
  Dtype* letternum_5_pred_data = letternum_5_pred_.mutable_cpu_data();
  const Dtype* let_5_data = bottom[6]->cpu_data();
  caffe_copy<Dtype>(bottom[6]->count(), let_5_data, letternum_5_pred_data);
  letternum_5_loss_layer_->Reshape(letternum_5_bottom_vec_, letternum_5_top_vec_);
  letternum_5_loss_layer_->Forward(letternum_5_bottom_vec_, letternum_5_top_vec_);

  /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  top[0]->mutable_cpu_data()[0] = 0;
  Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
  if(this->layer_param_.propagate_down(0)) {
    top[0]->mutable_cpu_data()[0] += 
          1*chinesecharcter_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(1)) {
    top[0]->mutable_cpu_data()[0] += 
          1*engcharcter_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(2)) {
    top[0]->mutable_cpu_data()[0] += 
          1*letternum_1_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(3)) {
    top[0]->mutable_cpu_data()[0] += 
          1*letternum_2_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(4)) {
    top[0]->mutable_cpu_data()[0] += 
          1*letternum_3_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(5)) {
    top[0]->mutable_cpu_data()[0] += 
          1*letternum_4_loss_.cpu_data()[0] / normalizer;
  }
  if(this->layer_param_.propagate_down(6)) {
    top[0]->mutable_cpu_data()[0] += 
          1*letternum_5_loss_.cpu_data()[0] / normalizer;
  }
  #if 0
  LOG(INFO)<<"top[0]->mutable_cpu_data()[0]: "<<top[0]->mutable_cpu_data()[0];
  #endif
}

template <typename Dtype>
void MulticcpdLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[7]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  // Back propagate on chinesecharcter_loss_ prediction.
  if (propagate_down[0]) {
    Dtype* chinese_bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), chinese_bottom_diff);
    vector<bool> chinese_propagate_down;
    // Only back propagate on prediction, not ground truth.
    chinese_propagate_down.push_back(true);
    chinese_propagate_down.push_back(false);
    chinesecharcter_loss_layer_->Backward(chinesecharcter_top_vec_, chinese_propagate_down,
                                chinesecharcter_bottom_vec_);
    // Scale gradient.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_scal(chinesecharcter_pred_.count(), loss_weight,
                chinesecharcter_pred_.mutable_cpu_diff());
        // The diff is already computed and stored.
    bottom[0]->ShareDiff(chinesecharcter_pred_);
  }

  // Back propagate on english prediction.
  if (propagate_down[1]) {
    Dtype* engcharcter_bottom_diff = bottom[1]->mutable_cpu_diff();
    caffe_set(bottom[1]->count(), Dtype(0), engcharcter_bottom_diff);
    vector<bool> english_propagate_down;
    // Only back propagate on prediction, not ground truth.
    english_propagate_down.push_back(true);
    english_propagate_down.push_back(false);
    engcharcter_loss_layer_->Backward(engcharcter_top_vec_, english_propagate_down,
                                engcharcter_bottom_vec_);
    // Scale gradient.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_scal(engcharcter_pred_.count(), loss_weight,
                engcharcter_pred_.mutable_cpu_diff());
    // Copy gradient back to bottom[1].
    
    bottom[1]->ShareDiff(engcharcter_pred_);
  }

  // Back propagate on letter prediction.
  if (propagate_down[2]) {
    Dtype* letter_1_bottom_diff = bottom[2]->mutable_cpu_diff();
    caffe_set(bottom[2]->count(), Dtype(0), letter_1_bottom_diff);
    vector<bool> letter_1_propagate_down;
    // Only back propagate on prediction, not ground truth.
    letter_1_propagate_down.push_back(true);
    letter_1_propagate_down.push_back(false);
    letternum_1_loss_layer_->Backward(letternum_1_top_vec_, letter_1_propagate_down,
                                letternum_1_bottom_vec_);
    // Scale gradient.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_scal(letternum_1_pred_.count(), loss_weight,
                letternum_1_pred_.mutable_cpu_diff());
    // Copy gradient back to bottom[2].
    bottom[2]->ShareDiff(letternum_1_pred_);
  }

  // Back propagate on letter 2 prediction.
  if (propagate_down[3]) {
    Dtype* letter_2_bottom_diff = bottom[3]->mutable_cpu_diff();
    caffe_set(bottom[3]->count(), Dtype(0), letter_2_bottom_diff);
    vector<bool> letter_2_propagate_down;
    // Only back propagate on prediction, not ground truth.
    letter_2_propagate_down.push_back(true);
    letter_2_propagate_down.push_back(false);
    letternum_2_loss_layer_->Backward(letternum_2_top_vec_, letter_2_propagate_down,
                                letternum_2_bottom_vec_);
    // Scale gradient.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_scal(letternum_2_pred_.count(), loss_weight,
                letternum_2_pred_.mutable_cpu_diff());
        // The diff is already computed and stored.
    bottom[3]->ShareDiff(letternum_2_pred_);
  }

// Back propagate on letter 3 prediction.
  if (propagate_down[4]) {
    Dtype* letter_3_bottom_diff = bottom[4]->mutable_cpu_diff();
    caffe_set(bottom[4]->count(), Dtype(0), letter_3_bottom_diff);
    vector<bool> letter_3_propagate_down;
    // Only back propagate on prediction, not ground truth.
    letter_3_propagate_down.push_back(true);
    letter_3_propagate_down.push_back(false);
    letternum_3_loss_layer_->Backward(letternum_3_top_vec_, letter_3_propagate_down,
                                letternum_3_bottom_vec_);
    // Scale gradient.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_scal(letternum_3_pred_.count(), loss_weight,
                letternum_3_pred_.mutable_cpu_diff());
        // The diff is already computed and stored.
    bottom[4]->ShareDiff(letternum_3_pred_);
  }

  // Back propagate on letter 4 prediction.
  if (propagate_down[5]) {
    Dtype* letter_4_bottom_diff = bottom[5]->mutable_cpu_diff();
    caffe_set(bottom[5]->count(), Dtype(0), letter_4_bottom_diff);
    vector<bool> letter_4_propagate_down;
    // Only back propagate on prediction, not ground truth.
    letter_4_propagate_down.push_back(true);
    letter_4_propagate_down.push_back(false);
    letternum_4_loss_layer_->Backward(letternum_4_top_vec_, letter_4_propagate_down,
                                letternum_4_bottom_vec_);
    // Scale gradient.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_scal(letternum_4_pred_.count(), loss_weight,
                letternum_4_pred_.mutable_cpu_diff());
    // The diff is already computed and stored.
    bottom[5]->ShareDiff(letternum_4_pred_);
  }

  // Back propagate on letter 5 prediction.
  if (propagate_down[6]) {
    Dtype* letter_5_bottom_diff = bottom[6]->mutable_cpu_diff();
    caffe_set(bottom[6]->count(), Dtype(0), letter_5_bottom_diff);
    vector<bool> letter_5_propagate_down;
    // Only back propagate on prediction, not ground truth.
    letter_5_propagate_down.push_back(true);
    letter_5_propagate_down.push_back(false);
    letternum_5_loss_layer_->Backward(letternum_5_top_vec_, letter_5_propagate_down,
                                letternum_5_bottom_vec_);
    // Scale gradient.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_scal(letternum_5_pred_.count(), loss_weight,
                letternum_5_pred_.mutable_cpu_diff());
    bottom[6]->ShareDiff(letternum_5_pred_);
  }
}

INSTANTIATE_CLASS(MulticcpdLossLayer);
REGISTER_LAYER_CLASS(MulticcpdLoss);

}  // namespace caffe
