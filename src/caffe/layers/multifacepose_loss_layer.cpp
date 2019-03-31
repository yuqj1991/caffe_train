#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multifacepose_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiFacePoseLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    if (this->layer_param_.propagate_down_size() == 0) {
        this->layer_param_.add_propagate_down(true);
        this->layer_param_.add_propagate_down(true);
        this->layer_param_.add_propagate_down(false);
    }
    const MultiFacePoseLossParameter& multifacepose_loss_param =
        this->layer_param_.multiface_pose_loss_param();
    multiface_loss_param_ = this->layer_param_.multiface_pose_loss_param();

    batch_size_ = bottom[0]->num();

    if (!this->layer_param_.loss_param().has_normalization() &&
        this->layer_param_.loss_param().has_normalize()) {
        normalization_ = this->layer_param_.loss_param().normalize() ?
                        LossParameter_NormalizationMode_VALID :
                        LossParameter_NormalizationMode_BATCH_SIZE;
    } else {
        normalization_ = this->layer_param_.loss_param().normalization();
    }
   
   /********************************************************************/
   // 21 face points face loss layer setup
    vector<int> loss_shape(1, 1);
    // Set up landmark loss layer.
    landmark_weight_ = multiface_loss_param.pose_weights();
    landmark_loss_type_ = multiface_loss_param.regs_face_contour_loss_type();
    // fake shape.
    vector<int> loc_shape(1, 1);
    loc_shape.push_back(42); //landmark_shape:{ 1, 21}x2
    landmark_pred_.Reshape(loc_shape); //loc_shape:{ }
    landmark_gt_.Reshape(loc_shape);
    landmark_bottom_vec_.push_back(&landmark_pred_);
    landmark_bottom_vec_.push_back(&landmark_gt_);
    landmark_loss_.Reshape(loss_shape);
    landmark_top_vec_.push_back(&landmark_loss_);
    if (landmark_loss_type_ == MultiFacePoseLossParameter_AttriLossType_L2) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_l2_loc");
        layer_param.set_type("EuclideanLoss");
        layer_param.add_loss_weight(landmark_weight_);
        landmark_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        landmark_loss_layer_->SetUp(landmark_bottom_vec_, landmark_top_vec_);
    } else if (landmark_loss_type_ == MultiFacePoseLossParameter_AttriLossType_SMOOTH_L1) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
        layer_param.set_type("SmoothL1Loss");
        layer_param.add_loss_weight(landmark_weight_);
        landmark_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        landmark_loss_layer_->SetUp(landmark_bottom_vec_, landmark_top_vec_);
    } else {
    LOG(FATAL) << "Unknown localization loss type.";
    }

    /*************************************************************/
    // facepose yaw pitch roll loss
    pose_loss_type_ = multiface_loss_param.regs_face_pose_loss_type();
    // fake shape.
    vector<int> loc_shape(1, 1);
    loc_shape.push_back(3); //facepose (yaw pitch roll)
    pose_pred_.Reshape(loc_shape); //loc_shape:{ }
    pose_gt_.Reshape(loc_shape);
    pose_bottom_vec_.push_back(&pose_pred_);
    pose_bottom_vec_.push_back(&pose_gt_);
    pose_loss_.Reshape(loss_shape);
    pose_top_vec_.push_back(&pose_loss_);
    if (pose_loss_type_ == MultiFacePoseLossParameter_AttriLossType_L2) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_l2_loc");
        layer_param.set_type("EuclideanLoss");
        layer_param.add_loss_weight(landmark_weight_);
        pose_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        pose_loss_layer_->SetUp(pose_bottom_vec_, pose_top_vec_);
    } else if (pose_loss_type_ == MultiFacePoseLossParameter_AttriLossType_SMOOTH_L1) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
        layer_param.set_type("SmoothL1Loss");
        layer_param.add_loss_weight(landmark_weight_);
        pose_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        pose_loss_layer_->SetUp(pose_bottom_vec_, pose_top_vec_);
    } else {
    LOG(FATAL) << "Unknown localization loss type.";
    }
}

template <typename Dtype>
void MultiFacePoseLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    int num_landmarks = bottom[0]->shape(1);  //bottom[0]: landmarks , 
    CHECK_EQ(num_landmarks, 42)<<"number of lanmarks point value coordinate must equal to 42";
}

template <typename Dtype>
void MultiFacePoseLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
    const Dtype* landmark_data = bottom[0]->cpu_data(); // landmark data
    const Dtype* pose_data = bottom[1]->cpu_data(); 
    const Dtype* label_data = bottom[2]->cpu_data();
    #if 0
    LOG(INFO)<< "loss compute start printf &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& num_gt_: "<<num_gt_;
    for(int ii=0; ii < num_gt_; ii++)
    {
    int id = ii*10;
    if (gt_data[id] == -1) {
        continue;
    }

    LOG(INFO) <<"LABEL batch_id: "<<gt_data[id]<<" anno_label: "<<gt_data[id+1]
                <<" anno.instance_id: "<<gt_data[id+2];
    LOG(INFO)  <<"LABEL bbox->xmin: "<<gt_data[id+3]<<" bbox->ymin: "<<gt_data[id+4]
                <<" bbox->xmax: "<<gt_data[id+5]<<" bbox->ymax: "<<gt_data[id+6]
                <<" bbox->blur: "<<gt_data[id+7]<<" bbox->occlusion: "<<gt_data[id+8];
    }
    LOG(INFO)<< "loss compute finished **************************************************** end ";

    #endif 
    /***************************************retrive all ground truth****************************************/
    // Retrieve all landmarks , gender, and glasses && headpose.
    map<int, AnnoFaceContourPoints > all_landmarks;
    map<int, AnnoFacePoseOritation > all_faceposes;
    all_landmarks.clear();
    for(int item_id = 0; item_id < batch_size_; item_id++){
        int idx = item_id*46;
        int id = label_data[idx];
        if(id == -1)
        {
            LOG(WARNING)<<"the item_id from each image, should not be the -1!!!1";
            continue;
        }
        AnnoFaceContourPoints facemark;
        AnnoFacePoseOritation facepose;
        facemark.point_x1(label_data[idx+1]);
        facemark.point_x2(label_data[idx+2]);
        facemark.point_x3(label_data[idx+3]);
        facemark.point_x4(label_data[idx+4]);
        facemark.point_x5(label_data[idx+5]);
        facemark.point_x6(label_data[idx+6]);
        facemark.point_x7(label_data[idx+7]);
        facemark.point_x8(label_data[idx+8]);
        facemark.point_x9(label_data[idx+9]);
        facemark.point_x10(label_data[idx+10]);
        facemark.point_x11(label_data[idx+11]);
        facemark.point_x12(label_data[idx+12]);
        facemark.point_x13(label_data[idx+13]);
        facemark.point_x14(label_data[idx+14]);
        facemark.point_x15(label_data[idx+15]);
        facemark.point_x16(label_data[idx+16]);
        facemark.point_x17(label_data[idx+17]);
        facemark.point_x18(label_data[idx+18]);
        facemark.point_x19(label_data[idx+19]);
        facemark.point_x20(label_data[idx+20]);
        facemark.point_x21(label_data[idx+21]);
        facemark.point_y1(label_data[idx+22]);
        facemark.point_y2(label_data[idx+23]);
        facemark.point_y3(label_data[idx+24]);
        facemark.point_y4(label_data[idx+25]);
        facemark.point_y5(label_data[idx+26]);
        facemark.point_y6(label_data[idx+27]);
        facemark.point_y7(label_data[idx+28]);
        facemark.point_y8(label_data[idx+29]);
        facemark.point_y9(label_data[idx+30]);
        facemark.point_y10(label_data[idx+31]);
        facemark.point_y11(label_data[idx+32]);
        facemark.point_y12(label_data[idx+33]);
        facemark.point_y13(label_data[idx+34]);
        facemark.point_y14(label_data[idx+35]);
        facemark.point_y15(label_data[idx+36]);
        facemark.point_y16(label_data[idx+37]);
        facemark.point_y17(label_data[idx+38]);
        facemark.point_y18(label_data[idx+39]);
        facemark.point_y19(label_data[idx+40]);
        facemark.point_y20(label_data[idx+41]);
        facemark.point_y21(label_data[idx+42]);
        facepose.set_yaw(label_data[idx+43]);
        facepose.set_pitch(label_data[idx+44]);
        facepose.set_roll(label_data[idx+45]);
        all_landmarks.insert(pair<int,AnnoFaceContourPoints>(item_id, facemark));
        all_poses.insert(pair<int,AnnoFacePoseOritation>(item_id, facepose));
    }
    CHECK_EQ(batch_size_, all_landmarks.size())<<"ground truth label size should match batch_size_";

    /***********************************************************************************/
    // Form data to pass on to landmark_loss_layer_.
    vector<int> landmark_shape(2);
    landmark_shape[0] = 1;
    landmark_shape[1] = batch_size_ * 42;
    landmark_pred_.Reshape(landmark_shape);
    landmark_gt_.Reshape(landmark_shape);
    landmark_pred_.CopyFrom(*bottom[0]);
    //Dtype* landmark_pred_data = landmark_pred_.mutable_cpu_data();
    Dtype* landmark_gt_data = landmark_gt_.mutable_cpu_data();
    int count =0;
    for(int ii = 0; ii< batch_size_; ii++)
    {
        AnnoFaceContourPoints face = all_landmarks[ii];
        landmark_gt_data[count*42] = face.point_x1() ;
        landmark_gt_data[count*42+1] = face.point_x2() ;
        landmark_gt_data[count*42+2] = face.point_x3() ;
        landmark_gt_data[count*42+3] = face.point_x4() ;
        landmark_gt_data[count*42+4] = face.point_x5() ;
        landmark_gt_data[count*42+5] = face.point_x6() ;
        landmark_gt_data[count*42+6] = face.point_x7() ;
        landmark_gt_data[count*42+7] = face.point_x8() ;
        landmark_gt_data[count*42+8] = face.point_x9() ;
        landmark_gt_data[count*42+9] = face.point_x10() ;
        landmark_gt_data[count*42+10] = face.point_x11() ;
        landmark_gt_data[count*42+11] = face.point_x12() ;
        landmark_gt_data[count*42+12] = face.point_x13() ;
        landmark_gt_data[count*42+13] = face.point_x14() ;
        landmark_gt_data[count*42+14] = face.point_x15() ;
        landmark_gt_data[count*42+15] = face.point_x16() ;
        landmark_gt_data[count*42+16] = face.point_x17() ;
        landmark_gt_data[count*42+17] = face.point_x18() ;
        landmark_gt_data[count*42+18] = face.point_x19() ;
        landmark_gt_data[count*42+19] = face.point_x20() ;
        landmark_gt_data[count*42+20] = face.point_x21() ;
        landmark_gt_data[count*42+21] = face.point_y1() ;
        landmark_gt_data[count*42+22] = face.point_y2() ;
        landmark_gt_data[count*42+23] = face.point_y3() ;
        landmark_gt_data[count*42+24] = face.point_y4() ;
        landmark_gt_data[count*42+25] = face.point_y5() ;
        landmark_gt_data[count*42+26] = face.point_y6() ;
        landmark_gt_data[count*42+27] = face.point_y7() ;
        landmark_gt_data[count*42+28] = face.point_y8() ;
        landmark_gt_data[count*42+29] = face.point_y9() ;
        landmark_gt_data[count*42+30] = face.point_y10() ;
        landmark_gt_data[count*42+31] = face.point_y11() ;
        landmark_gt_data[count*42+32] = face.point_y12() ;
        landmark_gt_data[count*42+33] = face.point_y13() ;
        landmark_gt_data[count*42+34] = face.point_y14() ;
        landmark_gt_data[count*42+35] = face.point_y15() ;
        landmark_gt_data[count*42+36] = face.point_y16() ;
        landmark_gt_data[count*42+37] = face.point_y17() ;
        landmark_gt_data[count*42+38] = face.point_y18() ;
        landmark_gt_data[count*42+39] = face.point_y19() ;
        landmark_gt_data[count*42+40] = face.point_y20() ;
        landmark_gt_data[count*42+41] = face.point_y21() ;
        ++count;
    }
    landmark_loss_layer_->Reshape(landmark_bottom_vec_, landmark_top_vec_);
    landmark_loss_layer_->Forward(landmark_bottom_vec_, landmark_top_vec_);

    /********************************************************************************/
    // Form data to pass on to pose_loss_layer_.
    // Reshape the pose confidence data.
    vector<int> pose_shape(2);
    pose_shape[0] = 1;
    pose_shape[1] = batch_size_ * 42;
    pose_pred_.Reshape(pose_shape);
    pose_gt_.Reshape(pose_shape);
    pose_pred_.CopyFrom(*bottom[1]);
    //Dtype* landmark_pred_data = landmark_pred_.mutable_cpu_data();
    Dtype* pose_gt_data = pose_gt_.mutable_cpu_data();
    count =0;
    for(int ii = 0; ii< batch_size_; ii++)
    {
        AnnoFacePoseOritation face = all_poses[ii];
        pose_gt_data[count*3] = face.yaw() ;
        pose_gt_data[count*3+1] = face.pitch() ;
        pose_gt_data[count*3+2] = face.roll() ;
        ++count;
    }
    pose_loss_layer_->Reshape(pose_bottom_vec_, pose_top_vec_);
    pose_loss_layer_->Forward(pose_bottom_vec_, pose_top_vec_);

    /**************************************sum loss value**************************************************/
    top[0]->mutable_cpu_data()[0] = 0;
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
    if (this->layer_param_.propagate_down(0)) {
    top[0]->mutable_cpu_data()[0] +=
        landmark_weight_ * landmark_loss_.cpu_data()[0] / normalizer;
    }
    if (this->layer_param_.propagate_down(1)) {
    top[0]->mutable_cpu_data()[0] += 
            pose_loss_.cpu_data()[0] / normalizer;
    }
    
    #if 0
    LOG(INFO)<<"num_matches_: "<<num_matches_<<" num_gtBoxes: "<<num_gt_<<" num_conf_: "<<num_conf_;
    LOG(INFO)<<" loc_loss_: "<< landmark_weight_ * loc_loss_.cpu_data()[0] / normalizer 
            <<" conf_loss_: "<<conf_loss_.cpu_data()[0] / normalizer
            <<" conf_blur_loss_: "<<0.5*conf_blur_loss_.cpu_data()[0] / normalizer
            <<" conf_occlussion_loss_: " << 0.5*conf_occlussion_loss_.cpu_data()[0] / normalizer;
    LOG(INFO)<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
    #endif
}

template <typename Dtype>
void MultiFacePoseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down,
const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[2]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }

    /*************************************************************************************/
    // Back propagate on landmark prediction.
    if (propagate_down[0]) {
        Dtype* mark_bottom_diff = bottom[0]->mutable_cpu_diff();
        caffe_set(bottom[0]->count(), Dtype(0), mark_bottom_diff);
        vector<bool> mark_propagate_down;
        // Only back propagate on prediction, not ground truth.
        mark_propagate_down.push_back(true);
        mark_propagate_down.push_back(false);
        landmark_loss_layer_->Backward(landmark_top_vec_, mark_propagate_down,
                                landmark_bottom_vec_);
        // Scale gradient.
        Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
            normalization_, batch_size_, 1, -1);
        Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
        caffe_scal(landmark_pred_.count(), loss_weight, landmark_pred_.mutable_cpu_diff());
        // Copy gradient back to bottom[0].
        const Dtype* landmark_pred_diff = landmark_pred_.cpu_diff();
        for (int ii = 0; ii < batch_size_; ++ii) {
            caffe_copy<Dtype>(42, landmark_pred_diff + ii * 42,
                                mark_bottom_diff + ii*42);
            mark_bottom_diff += bottom[0]->offset(1);
        }
    }

    /*************************************************************************************/
    // Back propagate on facepose loc prediction.
    if (propagate_down[1]) {
        Dtype* mark_bottom_diff = bottom[1]->mutable_cpu_diff();
        caffe_set(bottom[1]->count(), Dtype(0), mark_bottom_diff);
        vector<bool> mark_propagate_down;
        // Only back propagate on prediction, not ground truth.
        mark_propagate_down.push_back(true);
        mark_propagate_down.push_back(false);
        pose_loss_layer_->Backward(pose_top_vec_, mark_propagate_down,
                                pose_bottom_vec_);
        // Scale gradient.
        Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
            normalization_, batch_size_, 1, -1);
        Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
        caffe_scal(pose_pred_.count(), loss_weight, pose_pred_.mutable_cpu_diff());
        // Copy gradient back to bottom[0].
        const Dtype* pose_pred_diff = pose_pred_.cpu_diff();
        for (int ii = 0; ii < batch_size_; ++ii) {
            caffe_copy<Dtype>(3, pose_pred_diff + ii * 3,
                                mark_bottom_diff + ii*3);
            mark_bottom_diff += bottom[1]->offset(1);
        }
    }
}

INSTANTIATE_CLASS(MultiFacePoseLossLayer);
REGISTER_LAYER_CLASS(MultiFacePoseLoss);

}  // namespace caffe
