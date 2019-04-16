#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multiface_pose_loss_layer.hpp"
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
    landmark_weight_ = multifacepose_loss_param.pose_weights();
    landmark_loss_type_ = multifacepose_loss_param.regs_face_contour_loss_type();
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
    pose_loss_type_ = multifacepose_loss_param.regs_face_pose_loss_type();
    // fake shape.
    vector<int> pose_loc_shape(1, 1);
    pose_loc_shape.push_back(3); //facepose (yaw pitch roll)
    pose_pred_.Reshape(pose_loc_shape); //loc_shape:{ }
    pose_gt_.Reshape(pose_loc_shape);
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
    const Dtype* label_data = bottom[2]->cpu_data();
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
        facemark.mutable_point_1()->set_x(label_data[idx+1]);
        facemark.mutable_point_2()->set_x(label_data[idx+2]);
        facemark.mutable_point_3()->set_x(label_data[idx+3]);
        facemark.mutable_point_4()->set_x(label_data[idx+4]);
        facemark.mutable_point_5()->set_x(label_data[idx+5]);
        facemark.mutable_point_6()->set_x(label_data[idx+6]);
        facemark.mutable_point_7()->set_x(label_data[idx+7]);
        facemark.mutable_point_8()->set_x(label_data[idx+8]);
        facemark.mutable_point_9()->set_x(label_data[idx+9]);
        facemark.mutable_point_10()->set_x(label_data[idx+10]);
        facemark.mutable_point_11()->set_x(label_data[idx+11]);
        facemark.mutable_point_12()->set_x(label_data[idx+12]);
        facemark.mutable_point_13()->set_x(label_data[idx+13]);
        facemark.mutable_point_14()->set_x(label_data[idx+14]);
        facemark.mutable_point_15()->set_x(label_data[idx+15]);
        facemark.mutable_point_16()->set_x(label_data[idx+16]);
        facemark.mutable_point_17()->set_x(label_data[idx+17]);
        facemark.mutable_point_18()->set_x(label_data[idx+18]);
        facemark.mutable_point_19()->set_x(label_data[idx+19]);
        facemark.mutable_point_20()->set_x(label_data[idx+20]);
        facemark.mutable_point_21()->set_x(label_data[idx+21]);

        facemark.mutable_point_1()->set_y(label_data[idx+22]);
        facemark.mutable_point_2()->set_y(label_data[idx+23]);
        facemark.mutable_point_3()->set_y(label_data[idx+24]);
        facemark.mutable_point_4()->set_y(label_data[idx+25]);
        facemark.mutable_point_5()->set_y(label_data[idx+26]);
        facemark.mutable_point_6()->set_y(label_data[idx+27]);
        facemark.mutable_point_7()->set_y(label_data[idx+28]);
        facemark.mutable_point_8()->set_y(label_data[idx+29]);
        facemark.mutable_point_9()->set_y(label_data[idx+30]);
        facemark.mutable_point_10()->set_y(label_data[idx+31]);
        facemark.mutable_point_11()->set_y(label_data[idx+32]);
        facemark.mutable_point_12()->set_y(label_data[idx+33]);
        facemark.mutable_point_13()->set_y(label_data[idx+34]);
        facemark.mutable_point_14()->set_y(label_data[idx+35]);
        facemark.mutable_point_15()->set_y(label_data[idx+36]);
        facemark.mutable_point_16()->set_y(label_data[idx+37]);
        facemark.mutable_point_17()->set_y(label_data[idx+38]);
        facemark.mutable_point_18()->set_y(label_data[idx+39]);
        facemark.mutable_point_19()->set_y(label_data[idx+40]);
        facemark.mutable_point_20()->set_y(label_data[idx+41]);
        facemark.mutable_point_21()->set_y(label_data[idx+42]);
        
        facepose.set_yaw(label_data[idx+43]);
        facepose.set_pitch(label_data[idx+44]);
        facepose.set_roll(label_data[idx+45]);
        all_landmarks.insert(pair<int,AnnoFaceContourPoints>(item_id, facemark));
        all_faceposes.insert(pair<int,AnnoFacePoseOritation>(item_id, facepose));
    }
    CHECK_EQ(batch_size_, all_landmarks.size())<<"ground truth label size should match batch_size_";

    /***********************************************************************************/
    // Form data to pass on to landmark_loss_layer_.
    vector<int> landmark_shape(2);
    landmark_shape[0] = 1;
    landmark_shape[1] = batch_size_ * 42;
    landmark_pred_.Reshape(landmark_shape);
    landmark_gt_.Reshape(landmark_shape);
    Blob<Dtype> landmark_temp;
    landmark_temp.ReshapeLike(*(bottom[0]));
    landmark_temp.CopyFrom(*(bottom[0]));
    landmark_temp.Reshape(landmark_shape);
    landmark_pred_.CopyFrom(landmark_temp);
    Dtype* landmark_gt_data = landmark_gt_.mutable_cpu_data();
    for(int ii = 0; ii< batch_size_; ii++)
    {
        AnnoFaceContourPoints face = all_landmarks[ii];
        landmark_gt_data[ii*42] = face.point_1().x() ;
        landmark_gt_data[ii*42+1] = face.point_2().x()  ;
        landmark_gt_data[ii*42+2] = face.point_3().x() ;
        landmark_gt_data[ii*42+3] = face.point_4().x() ;
        landmark_gt_data[ii*42+4] = face.point_5().x() ;
        landmark_gt_data[ii*42+5] = face.point_6().x() ;
        landmark_gt_data[ii*42+6] = face.point_7().x() ;
        landmark_gt_data[ii*42+7] = face.point_8().x() ;
        landmark_gt_data[ii*42+8] = face.point_9().x() ;
        landmark_gt_data[ii*42+9] = face.point_10().x() ;
        landmark_gt_data[ii*42+10] = face.point_11().x() ;
        landmark_gt_data[ii*42+11] = face.point_12().x() ;
        landmark_gt_data[ii*42+12] = face.point_13().x() ;
        landmark_gt_data[ii*42+13] = face.point_14().x() ;
        landmark_gt_data[ii*42+14] = face.point_15().x() ;
        landmark_gt_data[ii*42+15] = face.point_16().x() ;
        landmark_gt_data[ii*42+16] = face.point_17().x() ;
        landmark_gt_data[ii*42+17] = face.point_18().x() ;
        landmark_gt_data[ii*42+18] = face.point_19().x() ;
        landmark_gt_data[ii*42+19] = face.point_20().x() ;
        landmark_gt_data[ii*42+20] = face.point_21().x() ;
        landmark_gt_data[ii*42+21] = face.point_1().y() ;
        landmark_gt_data[ii*42+22] = face.point_2().y() ;
        landmark_gt_data[ii*42+23] = face.point_3().y() ;
        landmark_gt_data[ii*42+24] = face.point_4().y() ;
        landmark_gt_data[ii*42+25] = face.point_5().y() ;
        landmark_gt_data[ii*42+26] = face.point_6().y() ;
        landmark_gt_data[ii*42+27] = face.point_7().y() ;
        landmark_gt_data[ii*42+28] = face.point_8().y() ;
        landmark_gt_data[ii*42+29] = face.point_9().y() ;
        landmark_gt_data[ii*42+30] = face.point_10().y() ;
        landmark_gt_data[ii*42+31] = face.point_11().y() ;
        landmark_gt_data[ii*42+32] = face.point_12().y() ;
        landmark_gt_data[ii*42+33] = face.point_13().y() ;
        landmark_gt_data[ii*42+34] = face.point_14().y() ;
        landmark_gt_data[ii*42+35] = face.point_15().y() ;
        landmark_gt_data[ii*42+36] = face.point_16().y() ;
        landmark_gt_data[ii*42+37] = face.point_17().y() ;
        landmark_gt_data[ii*42+38] = face.point_18().y() ;
        landmark_gt_data[ii*42+39] = face.point_19().y() ;
        landmark_gt_data[ii*42+40] = face.point_20().y() ;
        landmark_gt_data[ii*42+41] = face.point_21().y() ;
    }
    landmark_loss_layer_->Reshape(landmark_bottom_vec_, landmark_top_vec_);
    landmark_loss_layer_->Forward(landmark_bottom_vec_, landmark_top_vec_);

    /****************************************************************************************************/
    // Form data to pass on to pose_loss_layer_.
    // Reshape the pose confidence data.
    vector<int> pose_shape(2);
    pose_shape[0] = 1;
    pose_shape[1] = batch_size_ * 3;
    pose_pred_.Reshape(pose_shape);
    pose_gt_.Reshape(pose_shape);
    Blob<Dtype> pose_temp;
    pose_temp.ReshapeLike(*(bottom[1]));
    pose_temp.CopyFrom(*(bottom[1]));
    pose_temp.Reshape(pose_shape);
    pose_pred_.CopyFrom(pose_temp);
    //Dtype* landmark_pred_data = landmark_pred_.mutable_cpu_data();
    Dtype* pose_gt_data = pose_gt_.mutable_cpu_data();
    for(int ii = 0; ii< batch_size_; ii++)
    {
        AnnoFacePoseOritation face = all_faceposes[ii];
        pose_gt_data[ii*3] = face.yaw() ;
        pose_gt_data[ii*3+1] = face.pitch() ;
        pose_gt_data[ii*3+2] = face.roll() ;
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
        bottom[0]->ShareDiff(landmark_pred_);
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
        bottom[1]->ShareDiff(pose_pred_);
    }
}

INSTANTIATE_CLASS(MultiFacePoseLossLayer);
REGISTER_LAYER_CLASS(MultiFacePoseLoss);

}  // namespace caffe
