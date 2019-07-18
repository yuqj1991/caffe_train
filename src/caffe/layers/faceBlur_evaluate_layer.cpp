#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/faceBlur_evaluate_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {
template <typename Dtype>
void faceBlurEvaluateLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LOG(INFO)<<"faceBlurEvaluatLayer layerSetUp!";
}

template <typename Dtype>
void faceBlurEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(2, 1);
  top[0]->Reshape(top_shape);  
}

template <typename Dtype>
void faceBlurEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.), top_data);
  int batch_size = bottom[0]->num();
  vector<int> top_shape(2, 1);
  top_shape.push_back(batch_size);
  top_shape.push_back(2);
  top[0]->Reshape(top_shape);
  /**#####################################################**/
  int blur_precisive = 0;
  int occlu_precisive = 0;
  for(int ii = 0; ii<batch_size; ii++){
    const int pre_index = ii*2;
    const int gt_index = ii*2;
    const Dtype* cur_det_data = det_data + pre_index;
    const Dtype* cur_gt_data = gt_data + gt_index;
    if(cur_det_data[0]==cur_gt_data[0])
      blur_precisive=1;
    if(cur_det_data[1]==cur_gt_data[1])
      occlu_precisive = 1;
    top_data[ii*2 + 0] = blur_precisive;
    top_data[ii*2 + 1] = occlu_precisive;
  }
}

INSTANTIATE_CLASS(faceBlurEvaluateLayer);
REGISTER_LAYER_CLASS(faceBlurEvaluate);

}  // namespace caffe
