#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_faceBlur_output_layer.hpp"



namespace caffe {

template <typename Dtype>
void DetectionFaceBlurOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const DetectionOutputParameter& detection_output_param =
      this->layer_param_.detection_output_param();
  //CHECK(detection_output_param.has_num_blur()) << "Must specify num_blur";
  CHECK(detection_output_param.has_num_occlusion()) << "Must specify num_occlusion";
  //num_blur_ = detection_output_param.num_blur();
  num_occlusion_ = detection_output_param.num_occlusion();
}

template <typename Dtype>
void DetectionFaceBlurOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), num_occlusion_);
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  // Each row is a 9 dimension vector, which stores
  // [image_id, label, confidence, xmin, ymin, xmax, ymax, blur, occlussion]
  top_shape.push_back(2);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DetectionFaceBlurOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //const Dtype* blur_data = bottom[1]->cpu_data();
  const Dtype* occlu_data = bottom[0]->cpu_data();
  const int num = bottom[0]->num();

  vector<int> top_shape(2, 1);
  top_shape.push_back(num);
  top_shape.push_back(2);
  top[0]->Reshape(top_shape);
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set<Dtype>(top[0]->count(), -1, top_data);
  for (int i = 0; i < num; ++i) {
    //const int blur_index = i*num_blur_;
    const int occlu_index = i*num_occlusion_;
    //const Dtype* cur_blur_data= blur_data + blur_index;
    const Dtype* cur_occlu_data= occlu_data + occlu_index;
    int max_index = 0;
    Dtype temp = 0.f; 
    /*
    for(int ii =0;ii<num_blur_;ii++){
      //LOG(INFO)<<"cur_blur_data["<<ii<<"]: "<<cur_blur_data[ii];
      if(temp<cur_blur_data[ii]){
        max_index = ii;
        temp = cur_blur_data[ii];
      }
    }
    */
    top_data[i * 2] = 0;//max_index;
    temp = 0.f;
    max_index = 0;
    for(int ii =0;ii<num_occlusion_;ii++){
      //LOG(INFO)<<"cur_occlu_data["<<ii<<"]: "<<cur_occlu_data[ii];
      if(temp<cur_occlu_data[ii]){
        max_index = ii;
        temp = cur_occlu_data[ii];
      }
    }
    top_data[i * 2 + 1] = max_index;
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DetectionOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(DetectionFaceBlurOutputLayer);
REGISTER_LAYER_CLASS(DetectionFaceBlurOutput);

}  // namespace caffe
