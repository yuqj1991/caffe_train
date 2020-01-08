#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <math.h>
#include "caffe/layers/rf_box_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReceptiveFieldBoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ReceptiveFieldBoxParameter& anchor_box_param =
      this->layer_param_.receptive_box_param();
  
  receptive_field_center_start_ = anchor_box_param.receptive_field_center_start();
  receptive_field_stride_ = anchor_box_param.receptive_field_stride();
  receptive_field_size_ = anchor_box_param.receptive_field_size();

  num_priors_ += 1;
}

template <typename Dtype>
void ReceptiveFieldBoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  vector<int> top_shape(3, 1);
  // Since all images in a batch has same height and width, we only need to
  // generate one set of priors which can be shared across all images.
  top_shape[0] = 1;
  // 2 channels. First channel stores the mean of each prior coordinate.
  // Second channel stores the variance of each prior coordinate.
  top_shape[1] = 1;
  top_shape[2] = layer_width * layer_height * num_priors_ * 4;
  CHECK_GT(top_shape[2], 0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ReceptiveFieldBoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  int img_width, img_height;
  if (img_h_ == 0 || img_w_ == 0) {
    img_width = bottom[1]->width();
    img_height = bottom[1]->height();
  } else {
    img_width = img_w_;
    img_height = img_h_;
  }
  Dtype* top_data = top[0]->mutable_cpu_data();
  int idx = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = receptive_field_center_start_ + receptive_field_stride_ * w;
      float center_y = receptive_field_center_start_ + receptive_field_stride_ * h;
      float box_width, box_height;
      // first prior: aspect_ratio = 1, size = min_size
      box_width = box_height = receptive_field_size_;
      // xmin
      top_data[idx++] = (center_x ) / img_width;
      // ymin
      top_data[idx++] = (center_y ) / img_height;
      // xmax
      top_data[idx++] = (box_width) / img_width;
      // ymax
      top_data[idx++] = (box_height) / img_height;
    }
  }
}

INSTANTIATE_CLASS(ReceptiveFieldBoxLayer);
REGISTER_LAYER_CLASS(ReceptiveFieldBox);

}  // namespace caffe