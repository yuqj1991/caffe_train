#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/ccpdlp_evaluate_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {
template <typename Dtype>
void LpEvaluateLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const LpEvaluateParameter& lp_paramer = this->layer_param_.lp_evaluate_param();
  num_chinese_ = lp_paramer.num_chinese();
  num_english_ = lp_paramer.num_english();
  num_letter_ = lp_paramer.num_letter();
}

template <typename Dtype>
void LpEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(2, 1);
  top[0]->Reshape(top_shape);  
}

template <typename Dtype>
void LpEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.), top_data);
  int batch_size = bottom[0]->num();
  vector<int> top_shape(2, 1);
  top_shape.push_back(batch_size);
  top_shape.push_back(1);
  top[0]->Reshape(top_shape);
  /**#####################################################**/
  int correct_precisive = 0;
  for(int ii = 0; ii<batch_size; ii++){
    const int pre_index = ii*7;
    const int gt_index = ii*7;
    const Dtype* cur_det_data = det_data + pre_index;
    const Dtype* cur_gt_data = gt_data + gt_index;
    if(cur_det_data[0]==cur_gt_data[0]&&cur_det_data[1]==cur_gt_data[1]&&
      cur_det_data[2]==cur_gt_data[2]&&cur_det_data[3]==cur_gt_data[3]&&
      cur_det_data[4]==cur_gt_data[4]&&cur_det_data[5]==cur_gt_data[5]&&
      cur_det_data[6]==cur_gt_data[6])
      correct_precisive=1;
    top_data[ii] = correct_precisive;
    #if 0
    for(int jj=0; jj<9; jj++){
      LOG(INFO)<<"#####"<<all_face_prediction_attributes[ii][jj];
    }
    LOG(INFO)<<"gender_index: "<<gender_index<<" glasses_index: "<<glasses_index<<" headpose_index: "<<headpose_index;
    LOG(INFO)<<"=====gt_gender_index: "<<all_gt_face_attributes[ii][0]<<" gt glassesindex: "<<all_gt_face_attributes[ii][1]<<" gt headpose index: "<<all_gt_face_attributes[ii][2];
    #endif
  }
}

INSTANTIATE_CLASS(LpEvaluateLayer);
REGISTER_LAYER_CLASS(LpEvaluate);

}  // namespace caffe
