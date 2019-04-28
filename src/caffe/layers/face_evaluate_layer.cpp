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
  CHECK(face_paramer.has_facetype())
      << "Must provide facetype.";
  facetype_ = face_paramer.facetype();
}

template <typename Dtype>
void FaceEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (facetype_ == FaceEvaluateParameter_FaceType_FACE_5_TYPE){
    CHECK_EQ(num_facepoints_, 5)<< "this face points should be 5";
    vector<int> top_shape(4, 1);
    top[0]->Reshape(top_shape);
  }else if (facetype_ == FaceEvaluateParameter_FaceType_FACE_21_TYPE)
  {
    CHECK_EQ(num_facepoints_, 21)<< "this face points should be  21";
    vector<int> top_shape(4, 1);
    top[0]->Reshape(top_shape);
  }else if (facetype_ == FaceEvaluateParameter_FaceType_FACE_ANGLE)
    vector<int> top_shape(4, 1);
    top[0]->Reshape(top_shape);    
}

template <typename Dtype>
void FaceEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.), top_data);
  int batch_size = bottom[0]->num();
    
  if(facetype_ == FaceEvaluateParameter_FaceType_FACE_5_TYPE){ // evaluate 5 face point and face attributes
    map<int, vector<float> > all_prediction_face_points;
    map<int, vector<float> > all_gt_face_points; 
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
      float gender_temp=0.0, glasses_temp=0.0, headpose_temp=0.0;
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
  }else if (facetype_ == FaceEvaluateParameter_FaceType_FACE_21_TYPE){  // evaluate 21 face point and yaw pitch roll 
    map<int, vector<float> > all_prediction_face_points;
    map<int, vector<float> > all_gt_face_points;
    map<int, vector<float> >all_face_prediction_attributes;
    map<int, vector<int> > all_gt_face_attributes;
    for(int ii = 0; ii<batch_size; ii++){
      for(int jj =1; jj<43; jj++){
        all_gt_face_points[ii].push_back(gt_data[ii*46+jj]);
      }
      for(int jj =43; jj<46; jj++){
        all_gt_face_attributes[ii].push_back(gt_data[ii*46+jj]);
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
    float correct_precisive_yaw =0;
    float correct_precisive_pitch =0;
    float correct_precisive_roll =0;
    for(int ii = 0; ii<batch_size; ii++){
      for(int jj = 0; jj< num_facepoints_*2; jj++){
        distance_loss += pow((all_prediction_face_points[ii][jj]-all_gt_face_points[ii][jj]), 2);
      }
      correct_precisive_yaw += pow(std::abs(all_face_prediction_attributes[ii][0]*360- all_gt_face_attributes[ii][0]*360),2);
      correct_precisive_pitch += pow(std::abs(all_face_prediction_attributes[ii][1]*360 - all_gt_face_attributes[ii][1]*360),2);
      correct_precisive_roll += pow(std::abs(all_face_prediction_attributes[ii][2]*360 - all_gt_face_attributes[ii][2]*360),2);
    }
    top_data[0]=float(distance_loss/batch_size);
    top_data[1]=float(correct_precisive_yaw/batch_size);
    top_data[2]=float(correct_precisive_pitch/batch_size);
    top_data[3]=float(correct_precisive_roll/batch_size);
  }else if (facetype_ == FaceEvaluateParameter_FaceType_FACE_ANGLE){
    map<int, vector<float> > all_prediction_;
    map<int, vector<float> > all_gt_;
    for(int ii = 0; ii<batch_size; ii++){
      for(int jj =0; jj< 3; jj++){
        all_gt_[ii].push_back(gt_data[ii*3+jj]);
        all_prediction_[ii].push_back(det_data[ii*bottom[0]->channels()+jj]);
      }
    }
    /**#####################################################**/
    // face angle precision
    float correct_precisive_yaw =0;
    float correct_precisive_pitch =0;
    float correct_precisive_roll =0;
    for(int ii = 0; ii<batch_size; ii++){
      correct_precisive_yaw += pow(std::abs(all_prediction_[ii][0]*360- all_gt_[ii][0]*360),2);
      correct_precisive_pitch += pow(std::abs(all_prediction_[ii][1]*360 - all_gt_[ii][1]*360),2);
      correct_precisive_roll += pow(std::abs(all_prediction_[ii][2]*360 - all_gt_[ii][2]*360),2);
    }
    top_data[0]=float(0/batch_size);
    top_data[1]=float(correct_precisive_yaw/batch_size);
    top_data[2]=float(correct_precisive_pitch/batch_size);
    top_data[3]=float(correct_precisive_roll/batch_size);
  }
}

INSTANTIATE_CLASS(FaceEvaluateLayer);
REGISTER_LAYER_CLASS(FaceEvaluate);

}  // namespace caffe
