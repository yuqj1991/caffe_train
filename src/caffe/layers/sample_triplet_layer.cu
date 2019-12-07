#include "caffe/layers/sample_triplet_layer.hpp"

namespace caffe {

template <typename Dtype>
void SampleTripletLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

      this->Forward_cpu(bottom, top);
}

template <typename Dtype>
void SampleTripletLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}

INSTANTIATE_LAYER_GPU_FUNCS(SampleTripletLayer);

}
