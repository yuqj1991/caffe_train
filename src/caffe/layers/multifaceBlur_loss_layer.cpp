#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multifaceBlur_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiBlurLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    //this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
  }
  const MultiBoxLossParameter& multibox_loss_param =
      this->layer_param_.multibox_loss_param();
  multibox_loss_param_ = this->layer_param_.multibox_loss_param();
  batch_size_ = bottom[0]->num();
  // Get other parameters.
  //CHECK(multibox_loss_param.has_num_blur()) << "Must prodived num_blur";
  CHECK(multibox_loss_param.has_num_occlusion()) << "Must provide num_occlu";
  //num_blur_ = multibox_loss_param.num_blur();
  num_occlu_ = multibox_loss_param.num_occlusion();

  if (!this->layer_param_.loss_param().has_normalization() &&
    this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                    LossParameter_NormalizationMode_VALID :
                    LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  vector<int> loss_shape(1, 1);
  #if 0
  // Set up blur confidence loss layer.
  blur_loss_type_ = multibox_loss_param.conf_blur_loss_type();
  blur_bottom_vec_.push_back(&blur_pred_);
  blur_bottom_vec_.push_back(&blur_gt_);
  blur_loss_.Reshape(loss_shape);
  blur_top_vec_.push_back(&blur_loss_);
  if (blur_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_blur_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> blur_shape(1, 1);
    blur_gt_.Reshape(blur_shape);
    blur_shape.push_back(num_blur_);
    blur_pred_.Reshape(blur_shape);
    blur_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    blur_loss_layer_->SetUp(blur_bottom_vec_, blur_top_vec_);
  } else if (blur_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_blur_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> blur_shape(1, 1);
    blur_shape.push_back(num_blur_);
    blur_gt_.Reshape(blur_shape);
    blur_pred_.Reshape(blur_shape);
    blur_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    blur_loss_layer_->SetUp(blur_bottom_vec_, blur_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  #endif
  // Set up occlu confidence loss layer.
  occlu_loss_type_ = multibox_loss_param.conf_occlu_loss_type();
  occlu_bottom_vec_.push_back(&occlu_pred_);
  occlu_bottom_vec_.push_back(&occlu_gt_);
  occlu_loss_.Reshape(loss_shape);
  occlu_top_vec_.push_back(&occlu_loss_);
  if (occlu_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_occlu_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> occlu_shape(1, 1);
    occlu_gt_.Reshape(occlu_shape);
    occlu_shape.push_back(num_occlu_);
    occlu_pred_.Reshape(occlu_shape);
    occlu_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    occlu_loss_layer_->SetUp(occlu_bottom_vec_, occlu_top_vec_);
  } else if (occlu_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_occlu_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> occlu_shape(1, 1);
    occlu_shape.push_back(num_occlu_);
    occlu_gt_.Reshape(occlu_shape);
    occlu_pred_.Reshape(occlu_shape);
    occlu_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    occlu_loss_layer_->SetUp(occlu_bottom_vec_, occlu_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
}

template <typename Dtype>
void MultiBlurLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  int num_occlu = bottom[0]->shape(1); //bottom[0]: num_occlu;
  //int num_blur = bottom[1]->shape(1);  //bottom[1]: blur , 
  CHECK_EQ(num_occlu, num_occlu_)<<"number of num_occlu_ must match prototxt provided";
  //CHECK_EQ(num_blur, num_blur_)<<"number of num_blur_ point value must equal";
  CHECK_EQ(bottom[0]->count(), num_occlu_*batch_size_)<<"count must equal";
  //CHECK_EQ(bottom[1]->count(), num_blur_*batch_size_)<<"count must equal";
}

template <typename Dtype>
void MultiBlurLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* gt_data = bottom[2]->cpu_data();
  /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  vector<int> all_blur;
  vector<int> all_occlu;
  for(int item_id =0; item_id<batch_size_; item_id++){
    int idxg = item_id*2;
    all_blur.push_back(gt_data[idxg+0]);
    all_occlu.push_back(gt_data[idxg+1]);
  }
  #if 0
  LOG(INFO)<<"==================loss layer groundth======";
  LOG(INFO)<<"batch_size_: "<<batch_size_;
  for(int idx =0; idx<batch_size_; idx++){
    LOG(INFO) <<" blur: "<<all_blur[idx]
              <<" occlu: "<<all_occlu[idx];
  }
  LOG(INFO)<<"=================END=======================";
  #endif
  /*~~~~~~~~~~~~~~~~~~~~~~~occlu_loss_layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // conf occlu layer
  vector<int> occlu_shape;
  if (occlu_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    occlu_shape.push_back(batch_size_);
    occlu_gt_.Reshape(occlu_shape);
    occlu_shape.push_back(num_occlu_);
    occlu_pred_.Reshape(occlu_shape);
  } else if (occlu_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    //occlu_shape.push_back(1);
    occlu_shape.push_back(batch_size_);
    occlu_shape.push_back(num_occlu_);
    occlu_gt_.Reshape(occlu_shape);
    occlu_pred_.Reshape(occlu_shape);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  Dtype* occlu_gt_data = occlu_gt_.mutable_cpu_data();
  caffe_set(occlu_gt_.count(), Dtype(0), occlu_gt_data);
  for(int ii=0;ii<batch_size_; ii++){
    switch (occlu_loss_type_)
    {
    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
      occlu_gt_data[ii] = all_occlu[ii];
      break;
    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
      occlu_gt_data[ii * num_occlu_ + all_occlu[ii]] = 1;
      break;
    default:
      LOG(FATAL) << "Unknown conf loss type.";
    }
  }
  Dtype* occlu_pred_data = occlu_pred_.mutable_cpu_data();
  const Dtype* occlu_data = bottom[0]->cpu_data();
  caffe_copy<Dtype>(bottom[0]->count(), occlu_data, occlu_pred_data);
  occlu_loss_layer_->Reshape(occlu_bottom_vec_, occlu_top_vec_);
  occlu_loss_layer_->Forward(occlu_bottom_vec_, occlu_top_vec_);
  
  #if 0
  /*~~~~~~~~~~~~~~~~~~~~~~blur loss layer  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // Reshape the blur confidence data.
  vector<int> blur_shape;
  if (blur_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    blur_shape.push_back(batch_size_);
    blur_gt_.Reshape(blur_shape);
    blur_shape.push_back(num_blur_);
    blur_pred_.Reshape(blur_shape);
  } else if (blur_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    //blur_shape.push_back(1);
    blur_shape.push_back(batch_size_);
    blur_shape.push_back(num_blur_);
    blur_gt_.Reshape(blur_shape);
    blur_pred_.Reshape(blur_shape);
    /************************************************/
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  Dtype* blur_gt_data = blur_gt_.mutable_cpu_data();
  caffe_set(blur_gt_.count(), Dtype(0), blur_gt_data);
  for(int ii=0;ii<batch_size_; ii++){
    switch (blur_loss_type_)
    {
    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
      blur_gt_data[ii] = all_blur[ii];
      break;
    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
      blur_gt_data[ii * num_blur_ + all_blur[ii]] = 1;
      break;
    default:
      LOG(FATAL) << "Unknown conf loss type.";
    }
  }
  Dtype* blur_pred_data = blur_pred_.mutable_cpu_data();
  const Dtype* blur_data = bottom[1]->cpu_data();
  caffe_copy<Dtype>(bottom[1]->count(), blur_data, blur_pred_data);
  blur_loss_layer_->Reshape(blur_bottom_vec_, blur_top_vec_);
  blur_loss_layer_->Forward(blur_bottom_vec_, blur_top_vec_);
  #endif

  /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  top[0]->mutable_cpu_data()[0] = 0;
  Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
  /*if(this->layer_param_.propagate_down(0)) {
    top[0]->mutable_cpu_data()[0] += 
          blur_loss_.cpu_data()[0] / normalizer;
  }*/
  if(this->layer_param_.propagate_down(1)) {
    top[0]->mutable_cpu_data()[0] += 
          1*occlu_loss_.cpu_data()[0] / normalizer;
  }
  #if 0
  LOG(INFO)<<"top[0]->mutable_cpu_data()[0]: "<<top[0]->mutable_cpu_data()[0];
  #endif
}

template <typename Dtype>
void MultiBlurLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  // Back propagate on occlu prediction.
  if (propagate_down[0]) {
    Dtype* occlu_bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), occlu_bottom_diff);
    vector<bool> occlu_propagate_down;
    // Only back propagate on prediction, not ground truth.
    occlu_propagate_down.push_back(true);
    occlu_propagate_down.push_back(false);
    occlu_loss_layer_->Backward(occlu_top_vec_, occlu_propagate_down,
                                occlu_bottom_vec_);
    // Scale gradient.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_scal(occlu_pred_.count(), loss_weight,
                occlu_pred_.mutable_cpu_diff());
    // Copy gradient back to bottom[0].
    
    bottom[0]->ShareDiff(occlu_pred_);
  }
  #if 0
  // Back propagate on blur_loss_ prediction.
  if (propagate_down[1]) {
    Dtype* blur_bottom_diff = bottom[1]->mutable_cpu_diff();
    caffe_set(bottom[1]->count(), Dtype(0), blur_bottom_diff);
    vector<bool> blur_propagate_down;
    // Only back propagate on prediction, not ground truth.
    blur_propagate_down.push_back(true);
    blur_propagate_down.push_back(false);
    blur_loss_layer_->Backward(blur_top_vec_, blur_propagate_down,
                                blur_bottom_vec_);
    // Scale gradient.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_scal(blur_pred_.count(), loss_weight,
                blur_pred_.mutable_cpu_diff());
        // The diff is already computed and stored.
    bottom[1]->ShareDiff(blur_pred_);
  }
  #endif
}

INSTANTIATE_CLASS(MultiBlurLossLayer);
REGISTER_LAYER_CLASS(MultiBlurLoss);

}  // namespace caffe
