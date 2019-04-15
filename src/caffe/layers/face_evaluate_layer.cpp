#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/face_evaluate_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void FaceEvaluateLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const FaceEvaluateParameter& face_paramer = this->layer_param_.face_evaluate_param();
  num_gender_ = face_paramer.num_gender();
  num_glasses_ = face_paramer.num_glasses();
  num_headpose_ = face_paramer.num_headpose();
  num_facepoints_ = face_paramer.facepoints();
  if (num_gender_ >0 && num_glasses_>0 && num_headpose_>0){
    face_attributes_ =true;
  } 
  facetype_ = face_paramer.facetype();
}

template <typename Dtype>
void FaceEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (facetype_ == FaceEvaluateParameter_FaceType_FACE_5_TYPE){
    CHECK_EQ(num_facepoints_, 5)<< "this face points should be 5";
    vector<int> top_shape(4, 1);
    top[0]->Reshape(top_shape);
  }else
  {
    CHECK_EQ(num_facepoints_, 21)<< "this face points should be  21";
    vector<int> top_shape(2, 1);
    top[0]->Reshape(top_shape);
  }
        
}

template <typename Dtype>
void FaceEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.), top_data);
  int batch_size = bottom[0]->num();

  map<int, vector<float> > all_prediction_face_points;
  map<int, vector<float> > all_gt_face_points;
    
  if(face_attributes_){ // evaluate 5 face point and face attributes 
    map<int, vector<float> >all_face_prediction_attributes;
    map<int, vector<int> > all_gt_face_attributes;
    for(int ii = 0; ii<batch_size; ii++){
      for(int jj =1; jj<11; jj++){
        all_gt_face_points[ii].push_back(gt_data[ii*14+jj]);
      }
      for(int jj =11; jj<14; jj++){
        all_gt_face_attributes[ii].push_back(gt_data[ii*14+jj]);
      }
      for(int jj =0; jj< num_facepoints_*2; jj++){
        all_prediction_face_points[ii].push_back(det_data[ii*bottom[0]->channels()+jj]);
      }
      for(int jj =num_facepoints_*2; jj< bottom[0]->channels(); jj++){
        all_face_prediction_attributes[ii].push_back(det_data[ii*bottom[0]->channels()+jj]);
      }
    }
    /**#####################################################**/
    // face precision
    float distance_loss =0.0;
    int correct_precisive_gender =0;
    int correct_precisive_headpose =0;
    int correct_precisive_glasses =0;
    for(int ii = 0; ii<batch_size; ii++){
      for(int jj = 0; jj< num_facepoints_*2; jj++){
        distance_loss += pow((all_prediction_face_points[ii][jj]-all_gt_face_points[ii][jj]), 2);
      }
      int gender_index=0; 
      int glasses_index=0; 
      int headpose_index=0;
      float gender_temp=0.0, glasses_temp=0.0 ;
      float headpose_temp=0.0;
      for(int jj=0; jj<2; jj++){
        if(gender_temp<all_face_prediction_attributes[ii][jj]){
          gender_index = jj;
          gender_temp = all_face_prediction_attributes[ii][jj];
        }
        if(glasses_temp<all_face_prediction_attributes[ii][jj+2]){
          glasses_index = jj;
          glasses_temp = all_face_prediction_attributes[ii][jj+2];
        }
      }
      for(int jj=0; jj<5; jj++){
        if(headpose_temp<all_face_prediction_attributes[ii][jj+4]){
          headpose_index = jj;
          headpose_temp = all_face_prediction_attributes[ii][jj+4];
        }
      }
      if(all_gt_face_attributes[ii][0]==gender_index)
        correct_precisive_gender++;
      if(all_gt_face_attributes[ii][1]==glasses_index)
        correct_precisive_glasses++;
      if(all_gt_face_attributes[ii][2]==headpose_index)
        correct_precisive_headpose++;
    }
    top_data[0]=float(distance_loss/batch_size);
    top_data[1]=float(correct_precisive_gender/batch_size);
    top_data[2]=float(correct_precisive_glasses/batch_size);
    top_data[3]=float(correct_precisive_headpose/batch_size);
  }else{  // evaluate 21 face point and yaw pitch roll 

  }
}

INSTANTIATE_CLASS(FaceEvaluateLayer);
REGISTER_LAYER_CLASS(FaceEvaluate);

}  // namespace caffe
