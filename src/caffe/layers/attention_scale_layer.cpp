#include <cfloat>
#include <vector>

#include "caffe/layers/attention_scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AttentionScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2);
}

template <typename Dtype>
void AttentionScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->channels() == bottom[0]->channels());
  }
  CHECK_EQ(bottom[0]->channels(), bottom[1]->count());
  top[0]->ReshapeLike(*bottom[0]);
  temp_data_.Reshape(*bottom[0]);
}

template <typename Dtype>
void AttentionScaleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data_a = bottom[0]->cpu_data();
  const Dtype* bottom_data_b = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data_a, top_data);
  int batch_size_ = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  for(int b = 0; b < batch_size_; b++){
    for(int c = 0; c < channels; c++){
      caffe_scal(height * width, bottom_data_b[c], top_data + c * height * width);
    }
    top_data += top[0]->offset(1);
    bottom_data_b += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void AttentionScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* temp_diff = temp_data_.mutable_cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      if(i == 0){  
        caffe_div(count, top_data, bottom_data, bottom_diff);
      }else if(i == 1){
        const Dtype* bottom_data_a = bottom[0]->cpu_data();
        int batch_size_ = bottom[0]->num();
        int channels = bottom[0]->channels();
        int height = bottom[0]->height();
        int width = bottom[0]->width();
        for(int b = 0; b < batch_size_; b++){
          for(int c = 0; c < channels; c++){
            caffe_mul(height * width, bottom_data_a + c * height * width, top_diff + c * height * width, temp_diff + c * height * width);
            Dtype diff = caffe_cpu_asum(height * width, temp_diff + c * height * width);
            caffe_copy(1, &diff, bottom_diff + c);
          }
          bottom_diff += bottom[i]->offset(1);
          bottom_data_a += bottom[0]->offset(1);
          temp_diff += temp_data_.offset(1);
          top_diff += top[0]->offset(1);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AttentionScaleLayer);
#endif

INSTANTIATE_CLASS(AttentionScaleLayer);
REGISTER_LAYER_CLASS(AttentionScale);

}  // namespace caffe
