#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/faceEvaluateLayer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {
template <typename Dtype>
void FaceAttriEvaluateLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const FaceEvaluateParameter& face_paramer = this->layer_param_.face_evaluate_param();
  num_gender_ = face_paramer.num_gender();
  //num_glasses_ = face_paramer.num_glasses();
  num_facepoints_ = face_paramer.facepoints();
  CHECK(face_paramer.has_facetype())
      << "Must provide facetype.";
  facetype_ = face_paramer.facetype();
}

template <typename Dtype>
void FaceAttriEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (facetype_ == FaceEvaluateParameter_FaceType_FACEATTRIBUTE)
  {
    vector<int> top_shape(2, 1);
    top_shape.push_back(bottom[0]->num());
    top_shape.push_back(10);
    top[0]->Reshape(top_shape);
  }   
}

template <typename Dtype>
void FaceAttriEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.), top_data);
  int batch_size = bottom[0]->num();
    
  if (facetype_ == FaceEvaluateParameter_FaceType_FACEATTRIBUTE){
    map<int, vector<float> > all_prediction_face_points;
    map<int, vector<float> > all_gt_face_points; 
    map<int, vector<float> >all_face_prediction_attributes;
    map<int, vector<int> > all_gt_face_attributes;
    map<int, vector<float> > all_prediction_face_angle;
    map<int, vector<float> > all_gt_face_angle;
    map<int, vector<float>> batchImgShape;
    for(int ii = 0; ii<batch_size; ii++){
      /**********ground truth************/
      for(int jj =0; jj<10; jj++){
        all_gt_face_points[ii].push_back(gt_data[ii*16+jj]);
      }
      for(int jj =10; jj<13; jj++){
        all_gt_face_angle[ii].push_back(gt_data[ii*16+jj]);
      }
      for(int jj =13; jj<15; jj++){
        all_gt_face_attributes[ii].push_back(gt_data[ii*16+jj]);
      }
      for(int jj =15; jj<17; jj++){
        batchImgShape[ii].push_back(gt_data[ii*16+jj]);
      }
      /**********prediction************/
      for(int jj =0; jj< 5*2; jj++){
        all_prediction_face_points[ii].push_back(det_data[ii*bottom[0]->channels()+jj]);
      }
      for(int jj =5*2; jj< 13; jj++){
        all_prediction_face_angle[ii].push_back(det_data[ii*bottom[0]->channels()+jj]);
      }
      for(int jj =13; jj< 17; jj++){
        all_face_prediction_attributes[ii].push_back(det_data[ii*bottom[0]->channels()+jj]);
      }
    }
    /**#####################################################**/
    // face precision
    int correct_precisive_gender =0;
    int correct_precisive_glasses =0;
    float correct_precisive_yaw =0;
    float correct_precisive_pitch =0;
    float correct_precisive_roll =0;
    for(int ii = 0; ii<batch_size; ii++){
      Dtype Dis_diag = std::sqrt(pow(batchImgShape[ii][0], 2) + pow(batchImgShape[ii][1], 2));
      for(int jj = 0; jj< 5; jj++){
        double pow_x = pow(batchImgShape[ii][0]*(all_prediction_face_points[ii][jj]-all_gt_face_points[ii][jj]), 2);
        double pow_y = pow(batchImgShape[ii][1]*(all_prediction_face_points[ii][jj + 5]-all_gt_face_points[ii][jj + 5]), 2);
        top_data[ii*10 + jj] = std::sqrt(pow_x + pow_y) / Dis_diag;
      }
      int gender_index=0; 
      int glasses_index=0; 
      float gender_temp=0.0, glasses_temp=0.0;
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
      if(all_gt_face_attributes[ii][0]==gender_index)
        correct_precisive_gender=1;
      if(all_gt_face_attributes[ii][1]==glasses_index)
        correct_precisive_glasses=1;
      top_data[ii*10 + 5] = correct_precisive_gender;
      top_data[ii*10 + 6] = correct_precisive_glasses;
      correct_precisive_yaw = cos(double((all_prediction_face_angle[ii][0] / 180 - all_gt_face_angle[ii][0] / 180)* M_PI));
      correct_precisive_pitch = cos(double((all_prediction_face_angle[ii][1] / 180 - all_gt_face_angle[ii][1] / 180)* M_PI));
      correct_precisive_roll = cos(double((all_prediction_face_angle[ii][2] / 180 - all_gt_face_angle[ii][2] / 180)  * M_PI));
      top_data[ii* 10 + 7]=float(correct_precisive_yaw);
      top_data[ii* 10 + 8]=float(correct_precisive_pitch);
      top_data[ii* 10 + 9]=float(correct_precisive_roll);
    }
  }
}

INSTANTIATE_CLASS(FaceAttriEvaluateLayer);
REGISTER_LAYER_CLASS(FaceAttriEvaluate);

}  // namespace caffe
