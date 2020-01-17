#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_loss_layer_centernet.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterNetfocalSigmoidWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter sigmoid_param(this->layer_param_);
  sigmoid_param.set_type("Sigmoid");
  sigmoid_layer_ = LayerRegistry<Dtype>::CreateLayer(sigmoid_param);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(&prob_);
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  int pred_num_classes_channels = bottom[0]->channels();
  int labeled_num_classes_channels = bottom[1]->channels();

  CHECK_EQ(pred_num_classes_channels, labeled_num_classes_channels);

  alpha_ = 2.0f;
  gamma_ = 4.0f;
  iterations_ = 0;
}

template <typename Dtype>
void CenterNetfocalSigmoidWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  prob_.ReshapeLike(*(bottom[0]));
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  if (top.size() >= 2) {
    // sigmoid output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void CenterNetfocalSigmoidWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid prob values.
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  batch_ = bottom[0]->num();
  num_class_ = bottom[0]->channels();
  width_ = bottom[0]->width();
  height_ = bottom[0]->height();
  
  int count = 0;
  Dtype loss = Dtype(0.);
  Dtype negitive_loss = Dtype(0.), positive_loss = Dtype(0.);

  int dim = num_class_ * height_ * width_;
  int dimScale = height_ * width_; 

  for(int b = 0; b < batch_; ++b){
    for(int c = 0; c < num_class_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          int index = dim * b + c * dimScale + h * width_ + w;
          Dtype prob_a = prob_data[index];
          Dtype label_a = label[index];
          if( int(label_a) == 1){
            loss -= log(std::max(prob_a, Dtype(FLT_MIN))) * std::pow(1 -prob_a, alpha_);
            count++;
          }else if(label_a < Dtype(1)){
            loss -= log(std::max(1 - prob_a, Dtype(FLT_MIN))) * std::pow(prob_a, alpha_) *
                           std::pow(1 - label_a, gamma_);
            
          }
        }
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = (loss)/ count;
  #if 1
  if(iterations_%1000 == 0){
    LOG(INFO)<<"batch_: "<<batch_<<", num_class: "<<num_class_<<", height: "<<height_<<", width: "<<width_
    <<", loss: "<<loss<<", count: "<<count<<", classes total_loss: "<<top[0]->mutable_cpu_data()[0];
  }
  #endif
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
  iterations_++;
}

template <typename Dtype>
void CenterNetfocalSigmoidWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int count = 0;
    int dim = num_class_ * height_ * width_;
    int dimScale = height_ * width_;
    for(int b = 0; b < batch_; ++b){
      for(int c = 0; c < num_class_; ++c) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int index = dim * b + c * dimScale + h * width_ + w;
            Dtype prob_a = prob_data[index];
            Dtype label_a = label[index];
            if(int(label_a) == 1){
              bottom_diff[index] = std::pow(1 - prob_a, alpha_) * (alpha_ * log(std::max(prob_a, Dtype(FLT_MIN))) * prob_a - 
                        (1 - prob_a));
              count++;
            }else if(label_a < Dtype(1))
              bottom_diff[index] = std::pow(1 - prob_a, gamma_) * std::pow(prob_a, alpha_) *( - alpha_ * (1 - prob_a) * log(std::max(1 - prob_a, Dtype(FLT_MIN)))
                                  + prob_a);
          }
        }
        //prob_data += prob_.offset(0, 1);
        //label += bottom[1]->offset(0, 1);
        //bottom_diff += bottom[0]->offset(0, 1);
      }
    }
    Dtype loss_weight = top[0]->cpu_diff()[0] / count;
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(CenterNetfocalSigmoidWithLossLayer);
#endif

INSTANTIATE_CLASS(CenterNetfocalSigmoidWithLossLayer);
REGISTER_LAYER_CLASS(CenterNetfocalSigmoidWithLoss);

}  // namespace caffe
