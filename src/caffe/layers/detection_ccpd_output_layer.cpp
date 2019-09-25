#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_ccpd_output_layer.hpp"



namespace caffe {

template <typename Dtype>
void DetectionCcpdOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const LpDetParameter& detection_output_param =
      this->layer_param_.lp_det_param();
  CHECK(detection_output_param.has_num_chinese()) << "Must specify num_chinese";
  CHECK(detection_output_param.has_num_english()) << "Must specify num_eng";
  CHECK(detection_output_param.has_num_letter()) << "Must specify num_letter";
  num_chinese_ = detection_output_param.num_chinese();
  num_english_ = detection_output_param.num_english();
  num_letter_ = detection_output_param.num_letter();
}

template <typename Dtype>
void DetectionCcpdOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[1]->num(), bottom[2]->num());
  CHECK_EQ(bottom[2]->num(), bottom[3]->num());
  CHECK_EQ(bottom[3]->num(), bottom[4]->num());
  CHECK_EQ(bottom[4]->num(), bottom[5]->num());
  CHECK_EQ(bottom[5]->num(), bottom[6]->num());
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  // Each row is a 9 dimension vector, which stores
  // [image_id, label, confidence, xmin, ymin, xmax, ymax]
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DetectionCcpdOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* chi_data = bottom[0]->cpu_data();
  const Dtype* eng_data = bottom[1]->cpu_data();
  const Dtype* let1_data = bottom[2]->cpu_data();
  const Dtype* let2_data = bottom[3]->cpu_data();
  const Dtype* let3_data = bottom[4]->cpu_data();
  const Dtype* let4_data = bottom[5]->cpu_data();
  const Dtype* let5_data = bottom[6]->cpu_data();
  const int num = bottom[0]->num();

  vector<int> top_shape(2, 1);
  top_shape.push_back(num);
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set<Dtype>(top[0]->count(), -1, top_data);
  for (int i = 0; i < num; ++i) {
    const int chi_index = i*num_chinese_;
    const int eng_index = i*num_english_;
    const int letter_index = i*num_letter_;
    const Dtype* cur_chi_data= chi_data + chi_index;
    const Dtype* cur_eng_data= eng_data + eng_index;
    const Dtype* cur_let_1_data= let1_data + letter_index;
    const Dtype* cur_let_2_data= let2_data + letter_index;
    const Dtype* cur_let_3_data= let3_data + letter_index;
    const Dtype* cur_let_4_data= let4_data + letter_index;
    const Dtype* cur_let_5_data= let5_data + letter_index;
    int max_index = 0;
    Dtype temp = 0.f; 
    for(int ii =0;ii<num_chinese_;ii++){
      //LOG(INFO)<<"cur_chi_data["<<ii<<"]: "<<cur_chi_data[ii];
      if(temp<cur_chi_data[ii]){
        max_index = ii;
        temp = cur_chi_data[ii];
      }
    }
    top_data[i * 7] = max_index;
    temp = 0.f;
    max_index = 0;
    for(int ii =0;ii<num_english_;ii++){
      //LOG(INFO)<<"cur_eng_data["<<ii<<"]: "<<cur_eng_data[ii];
      if(temp<cur_eng_data[ii]){
        max_index = ii;
        temp = cur_eng_data[ii];
      }
    }
    top_data[i * 7 + 1] = max_index;
    temp = 0.f;
    max_index = 0;
    for(int ii =0;ii<num_letter_;ii++){
      //LOG(INFO)<<"cur_let_1_data["<<ii<<"]: "<<cur_let_1_data[ii];
      if(temp<cur_let_1_data[ii]){
        max_index = ii;
        temp = cur_let_1_data[ii];
      }
    }
    top_data[i * 7 + 2] = max_index;
    temp = 0.f;
    max_index = 0;
    for(int ii =0;ii<num_letter_;ii++){
      //LOG(INFO)<<"cur_let_2_data["<<ii<<"]: "<<cur_let_2_data[ii];
      if(temp<cur_let_2_data[ii]){
        max_index = ii;
        temp = cur_let_2_data[ii];
      }
    }
    top_data[i * 7 + 3] = max_index;
    temp = 0.f;
    max_index = 0;
    for(int ii =0;ii<num_letter_;ii++){
      //LOG(INFO)<<"cur_let_3_data["<<ii<<"]: "<<cur_let_3_data[ii];
      if(temp<cur_let_3_data[ii]){
        max_index = ii;
        temp = cur_let_3_data[ii];
      }
    }
    top_data[i * 7 + 4] = max_index;
    temp = 0.f;
    max_index = 0;
    for(int ii =0;ii<num_letter_;ii++){
      //LOG(INFO)<<"cur_let_4_data["<<ii<<"]: "<<cur_let_4_data[ii];
      if(temp<cur_let_4_data[ii]){
        max_index = ii;
        temp = cur_let_4_data[ii];
      }
    }
    top_data[i * 7 + 5] = max_index;
    temp = 0.f;
    max_index = 0;
    for(int ii =0;ii<num_letter_;ii++){
      //LOG(INFO)<<"cur_let_5_data["<<ii<<"]: "<<cur_let_5_data[ii];
      if(temp<cur_let_5_data[ii]){
        max_index = ii;
        temp = cur_let_5_data[ii];
      }
    }
    top_data[i * 7 + 6] = max_index;
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DetectionCcpdOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(DetectionCcpdOutputLayer);
REGISTER_LAYER_CLASS(DetectionCcpdOutput);

}  // namespace caffe
