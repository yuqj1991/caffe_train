#include "caffe/layers/triplet_loss_layer.hpp"
#include "math.h"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  batch_size_ = bottom[0]->num();
  feature_dim_ = bottom[0]->count(1);
  triplet_num_ = bottom[1]->num();
  diff_na_.Reshape(feature_dim_, 1, 1, 1);
  diff_pa_.Reshape(feature_dim_, 1, 1, 1);
  diff_np_.Reshape(feature_dim_, 1, 1, 1);
  bottom_diff_.Reshape(batch_size_, feature_dim_, 1, 1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  vector<int> loss_shape(1);
  loss_shape[0]=1;
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::ComputeDiff_cpu(const Dtype *x_1,
  const Dtype *x_2, const Dtype x_1_norm, const Dtype x_2_norm,
  const Dtype inner_val, Dtype *x_1_diff) {
  caffe_cpu_scale(feature_dim_, Dtype(1) / (x_1_norm * x_2_norm),
      x_2, x_1_diff);
  Dtype x_1_norm_cubic = x_1_norm * x_1_norm * x_1_norm;
  caffe_cpu_axpby(feature_dim_, -inner_val / (x_1_norm_cubic * x_2_norm),
      x_1, Dtype(1), x_1_diff);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff_.mutable_cpu_data());

  Dtype loss = 0;
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  for (int i = 0; i < triplet_num_; ++i) {
    int a_idx = bottom[1]->cpu_data()[i * 3 ];
    int p_idx = bottom[1]->cpu_data()[i * 3 + 1];
    int n_idx = bottom[1]->cpu_data()[i * 3 + 2];
    const Dtype *a_pointer = bottom[0]->cpu_data() + a_idx * feature_dim_;
    const Dtype *p_pointer = bottom[0]->cpu_data() + p_idx * feature_dim_;
    const Dtype *n_pointer = bottom[0]->cpu_data() + n_idx * feature_dim_;
    /*************以下为自己添加的*********************/
    Dtype dist_ap = 0.f, dist_an = 0.f;
    for(int i = 0; i< feature_dim_; i++){
      float diff_apos = std::pow(float(a_pointer[i])- float(p_pointer[i]), 2);
      float diff_aneg = std::pow(float(a_pointer[i] )- float(n_pointer[i]), 2);
      dist_ap += diff_apos;
      dist_an +=diff_aneg;
    }
    //LOG(INFO)<<"dist_ap: "<<dist_ap<<" dist_an: "<<dist_an;
    /***********************************************/
    if (dist_ap - dist_an + margin > 0) {
      loss += dist_ap + margin - dist_an;
    }
    /*********************以下是我自己做的**********************/
    // backword a
    caffe_sub(
      feature_dim_,
      n_pointer,  // n
      p_pointer,  // p
      diff_np_.mutable_cpu_data());  // n_i - p_i;
    caffe_cpu_axpby(
      feature_dim_,
      Dtype(2.0),
      diff_np_.mutable_cpu_data(),
      Dtype(0.0),
      bottom_diff_.mutable_cpu_data() + (a_idx * feature_dim_));
    // backward p
    caffe_sub(
      feature_dim_,
      p_pointer,  // p
      a_pointer,  // a
      diff_pa_.mutable_cpu_data());  // p_i - a_i;
    caffe_cpu_axpby(
      feature_dim_,
      Dtype(2.0),
      diff_pa_.mutable_cpu_data(),
      Dtype(0.0),
      bottom_diff_.mutable_cpu_data() + (p_idx * feature_dim_)); 
    // backward n
    caffe_sub(
      feature_dim_,
      a_pointer,  // a
      p_pointer,  // n
      diff_na_.mutable_cpu_data());  // a_i - n_i;
    caffe_cpu_axpby(
      feature_dim_,
      Dtype(2.0),
      diff_na_.mutable_cpu_data(),
      Dtype(0.0),
      bottom_diff_.mutable_cpu_data() + (n_idx * feature_dim_)); 
    /*********************以上是我自己做的**********************/
  }
  Dtype scalar = Dtype(1) / triplet_num_;
  top[0]->mutable_cpu_data()[0] = loss * scalar;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype scalar = top[0]->cpu_diff()[0] / triplet_num_;
    caffe_cpu_scale(bottom_diff_.count(), scalar, bottom_diff_.cpu_data(),
        bottom[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}
