#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_centerGrid_output_layer.hpp"

namespace caffe {

bool GridCompareScore(CenterNetInfo a, CenterNetInfo b){
    return a.score() > b.score();
}

template <typename Dtype>
void CenterGridOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  bottom_size_ = bottom.size();
  const DetectionOutputParameter& detection_output_param =
      this->layer_param_.detection_output_param();
  CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes";
  num_classes_ = detection_output_param.num_classes();
  if(detection_output_param.bias_scale_size() > 0){
    CHECK_EQ(bottom_size_, detection_output_param.bias_scale_size());
    CHECK_EQ(bottom_size_, detection_output_param.down_ratio_size());
    for(int i = 0; i < detection_output_param.bias_scale_size(); i++){
      anchor_scale_.push_back(detection_output_param.bias_scale(i));
      downRatio_.push_back(detection_output_param.down_ratio(i));
    }
  }
  keep_top_k_ = detection_output_param.keep_top_k();
  confidence_threshold_ = detection_output_param.has_confidence_threshold() ?
      detection_output_param.confidence_threshold() : -FLT_MAX;
  ignore_thresh_ = detection_output_param.ignore_thresh();
}

template <typename Dtype>
void CenterGridOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[0]->channels(), 4 + 1 + num_classes_);
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  // Each row is a 7 dimension vector, which stores
  // [image_id, label, confidence, xmin, ymin, xmax, ymax]
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void CenterGridOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  bottom_size_ = bottom.size();
  results_.clear();
  for(int t = 0; t < bottom_size_; t++){
    Dtype *channel_pred_data = bottom[t]->mutable_cpu_data();
    const int output_height = bottom[0]->height();
    const int output_width = bottom[0]->width();
    num_ = bottom[t]->num();
    int num_channels = bottom[t]->channels();
    int net_width = output_width * downRatio_[t];
    int net_height = output_height * downRatio_[t];
    GetCenterGridObjectResult(num_, num_channels, num_classes_,
                          output_width, output_height, 
                          net_width, net_height,
                          channel_pred_data, anchor_scale_[t], confidence_threshold_, &results_);
    
  }
  int num_kept = 0;

  // nms 去除多余的框
  std::map<int, vector<CenterNetInfo > > ::iterator iter;
  for(iter = results_.begin(); iter != results_.end(); iter++){
    std::sort(iter->second.begin(), iter->second.end(), GridCompareScore);
    std::vector<CenterNetInfo> temp_result = iter->second;
    std::vector<CenterNetInfo> nms_result;
    center_nms(temp_result, &nms_result, ignore_thresh_);
    int num_det = nms_result.size();
    if(keep_top_k_ > 0 && num_det > keep_top_k_){
      std::sort(nms_result.begin(), nms_result.end(), GridCompareScore);
      nms_result.resize(keep_top_k_);
      num_kept += keep_top_k_;
    }else{
      num_kept += num_det;
    }
    iter->second.clear();
    for(unsigned ii = 0; ii < nms_result.size(); ii++){
      iter->second.push_back(nms_result[ii]);
    }
  }
  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(7);
  Dtype* top_data;
  if (num_kept == 0) {
    top_shape[2] = num_;
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), -1, top_data);
    // Generate fake results per image.
    for (int i = 0; i < num_; ++i) {
      top_data[0] = i;
      top_data += 7;
    }
  } else {
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
  }
  // 保存生成新的结果
  int count = 0;
  for(int i = 0; i < num_; i++){
    if(results_.find(i) != results_.end()){
      std::vector<CenterNetInfo > result_temp = results_.find(i)->second;
      LOG(INFO)<<"batch_id "<<i << " detection results: "<<result_temp.size();
      for(unsigned j = 0; j < result_temp.size(); ++j){
        top_data[count * 7] = i;
        top_data[count * 7 + 1] = result_temp[j].class_id() + 1;
        top_data[count * 7 + 2] = result_temp[j].score();
        top_data[count * 7 + 3] = result_temp[j].xmin();
        top_data[count * 7 + 4] = result_temp[j].ymin();
        top_data[count * 7 + 5] = result_temp[j].xmax();
        top_data[count * 7 + 6] = result_temp[j].ymax();
        LOG(INFO)<<"class: "<<top_data[count * 7 + 1]<<", "<<result_temp[j].class_id() + 1
                 <<", center_x: "<< (result_temp[j].xmin() + result_temp[j].xmax()) / 2
                 <<", center_y: "<< (result_temp[j].ymin() + result_temp[j].ymax()) / 2
                 <<", width: "<< result_temp[j].xmax() - result_temp[j].xmin()
                 <<", height: "<< result_temp[j].ymax() - result_temp[j].ymin();
        ++count;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(CenterGridOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(CenterGridOutputLayer);
REGISTER_LAYER_CLASS(CenterGridOutput);

}  // namespace caffe
