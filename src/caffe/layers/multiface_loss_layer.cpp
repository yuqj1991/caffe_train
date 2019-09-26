#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multiface_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiFaceLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    if (this->layer_param_.propagate_down_size() == 0) {
        this->layer_param_.add_propagate_down(true);
        this->layer_param_.add_propagate_down(true);
        this->layer_param_.add_propagate_down(true);
        this->layer_param_.add_propagate_down(true);
        this->layer_param_.add_propagate_down(false);
    }
    const MultiFaceAttriLossParameter& multiface_loss_param =
        this->layer_param_.multiface_loss_param();
    multiface_loss_param_ = this->layer_param_.multiface_loss_param();

    batch_size_ = bottom[0]->num();
    // Get other parameters.
    CHECK_EQ(multiface_loss_param.num_gender(), 2) << "Must provide num_gender, and the num_gender must is 2.";
    CHECK_EQ(multiface_loss_param.num_glasses(), 2) << "Must prodived num_glasses, and the num_glasses must is 2";
    num_gender_ = multiface_loss_param.num_gender();
    num_glasses_ = multiface_loss_param.num_glasses();

    if (!this->layer_param_.loss_param().has_normalization() &&
        this->layer_param_.loss_param().has_normalize()) {
        normalization_ = this->layer_param_.loss_param().normalize() ?
                        LossParameter_NormalizationMode_VALID :
                        LossParameter_NormalizationMode_BATCH_SIZE;
    } else {
        normalization_ = this->layer_param_.loss_param().normalization();
    }

    vector<int> loss_shape(1, 1);
    /*****************************************************************************************/
    // Set up landmark loss layer.
    landmark_weight_ = multiface_loss_param.mark_weight();
    landmark_loss_type_ = multiface_loss_param.mark_loss_type();
    // fake shape.
    vector<int> loc_shape(1, 1);
    loc_shape.push_back(10); //landmark_shape:{ 1, 5}
    landmark_pred_.Reshape(loc_shape); //loc_shape:{ }
    landmark_gt_.Reshape(loc_shape);
    landmark_bottom_vec_.push_back(&landmark_pred_);
    landmark_bottom_vec_.push_back(&landmark_gt_);
    landmark_loss_.Reshape(loss_shape);
    landmark_top_vec_.push_back(&landmark_loss_);
    if (landmark_loss_type_ == MultiFaceAttriLossParameter_MarkLossType_L2) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_l2_loc");
        layer_param.set_type("EuclideanLoss");
        layer_param.add_loss_weight(landmark_weight_);
        landmark_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        landmark_loss_layer_->SetUp(landmark_bottom_vec_, landmark_top_vec_);
    } else if (landmark_loss_type_ == MultiFaceAttriLossParameter_MarkLossType_SMOOTH_L1) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
        layer_param.set_type("SmoothL1Loss");
        layer_param.add_loss_weight(landmark_weight_);
        landmark_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        landmark_loss_layer_->SetUp(landmark_bottom_vec_, landmark_top_vec_);
    } else {
    LOG(FATAL) << "Unknown localization loss type.";
    }

    /*****************************************************************************************/
    // Set up angle loss layer.
    angle_weight_ = multiface_loss_param.angle_weight();
    angle_loss_type_ = multiface_loss_param.angle_loss_type();
    // fake shape.
    vector<int> angle_shape(1, 1);
    angle_shape.push_back(3);
    angle_pred_.Reshape(angle_shape); //loc_shape:{ }
    angle_gt_.Reshape(angle_shape);
    angle_bottom_vec_.push_back(&angle_pred_);
    angle_bottom_vec_.push_back(&angle_gt_);
    angle_loss_.Reshape(loss_shape);
    angle_top_vec_.push_back(&angle_loss_);
    if (angle_loss_type_ == MultiFaceAttriLossParameter_MarkLossType_L2) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_l2_loc");
        layer_param.set_type("EuclideanLoss");
        layer_param.add_loss_weight(angle_weight_);
        angle_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        angle_loss_layer_->SetUp(angle_bottom_vec_, angle_top_vec_);
    } else if (angle_loss_type_ == MultiFaceAttriLossParameter_MarkLossType_SMOOTH_L1) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
        layer_param.set_type("SmoothL1Loss");
        layer_param.add_loss_weight(angle_weight_);
        angle_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        angle_loss_layer_->SetUp(angle_bottom_vec_, angle_top_vec_);
    } else {
    LOG(FATAL) << "Unknown localization loss type.";
    }

    /*****************************************************************************************/
    // Set up gender confidence loss layer.
    gender_weight_ = multiface_loss_param.gender_weight();
    gender_loss_type_ = multiface_loss_param.conf_gender_loss_type();
    gender_bottom_vec_.push_back(&gender_pred_);
    gender_bottom_vec_.push_back(&gender_gt_);
    gender_loss_.Reshape(loss_shape);
    gender_top_vec_.push_back(&gender_loss_);
    if (gender_loss_type_ == MultiFaceAttriLossParameter_AttriLossType_SOFTMAX) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_softmax_gender_conf");
        layer_param.set_type("SoftmaxWithLoss");
        layer_param.add_loss_weight(Dtype(1.));
        layer_param.mutable_loss_param()->set_normalization(
            LossParameter_NormalizationMode_NONE);
        SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
        softmax_param->set_axis(1);
        // Fake reshape.
        vector<int> gender_shape(1, 1);
        gender_gt_.Reshape(gender_shape);
        gender_shape.push_back(num_gender_);
        gender_pred_.Reshape(gender_shape);
        gender_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        gender_loss_layer_->SetUp(gender_bottom_vec_, gender_top_vec_);
    } else if (gender_loss_type_ == MultiFaceAttriLossParameter_AttriLossType_LOGISTIC) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_logistic_gender_conf");
        layer_param.set_type("SigmoidCrossEntropyLoss");
        layer_param.add_loss_weight(Dtype(1.));
        // Fake reshape.
        vector<int> gender_shape(1, 1);
        gender_shape.push_back(num_gender_);
        gender_gt_.Reshape(gender_shape);
        gender_pred_.Reshape(gender_shape);
        gender_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        gender_loss_layer_->SetUp(gender_bottom_vec_, gender_top_vec_);
    } else {
    LOG(FATAL) << "Unknown face attributes gender loss type.";
    }

    /*****************************************************************************************/
    // Set up glasses confidence loss layer.
    glasses_weight_ = multiface_loss_param.glass_weight();
    glasses_loss_type_ = multiface_loss_param.conf_glasses_loss_type();
    glasses_bottom_vec_.push_back(&glasses_pred_);
    glasses_bottom_vec_.push_back(&glasses_gt_);
    glasses_loss_.Reshape(loss_shape);
    glasses_top_vec_.push_back(&glasses_loss_);
    if (glasses_loss_type_ == MultiFaceAttriLossParameter_AttriLossType_SOFTMAX) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_softmax_glasses_conf");
        layer_param.set_type("SoftmaxWithLoss");
        layer_param.add_loss_weight(Dtype(1.));
        layer_param.mutable_loss_param()->set_normalization(
            LossParameter_NormalizationMode_NONE);
        SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
        softmax_param->set_axis(1);
        // Fake reshape.
        vector<int> glasses_shape(1, 1);
        glasses_gt_.Reshape(glasses_shape);
        glasses_shape.push_back(num_glasses_);
        glasses_pred_.Reshape(glasses_shape);
        glasses_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        glasses_loss_layer_->SetUp(glasses_bottom_vec_, glasses_top_vec_);
    } else if (glasses_loss_type_ == MultiFaceAttriLossParameter_AttriLossType_LOGISTIC) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_logistic_glasses_conf");
        layer_param.set_type("SigmoidCrossEntropyLoss");
        layer_param.add_loss_weight(Dtype(1.));
        // Fake reshape.
        vector<int> glasses_shape(1, 1);
        glasses_shape.push_back(num_glasses_);
        glasses_gt_.Reshape(glasses_shape);
        glasses_pred_.Reshape(glasses_shape);
        glasses_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        glasses_loss_layer_->SetUp(glasses_bottom_vec_, glasses_top_vec_);
    } else {
    LOG(FATAL) << "Unknown face attributes glasses loss type.";
    }
}

template <typename Dtype>
void MultiFaceLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    int num_landmarks = bottom[0]->shape(1);  //bottom[0]: landmarks , 
    int num_gender = bottom[2]->shape(1); //bottom[1]: num_gender;
    int num_glasses = bottom[3]->shape(1); // bottom[2]: num_glasses;
    CHECK_EQ(num_landmarks, 10)<<"number of lanmarks point value must equal to 10";
    CHECK_EQ(num_gender_, num_gender)<<"number of gender must match prototxt provided";
    CHECK_EQ(num_glasses_, num_glasses)<<"number of glasses must match prototxt provided";
}

template <typename Dtype>
void MultiFaceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
    const Dtype* label_data = bottom[4]->cpu_data();
 
    /***********************************************************************************/
    // Retrieve all landmarks , gender, and glasses.***********.
    vector<int> all_gender;
    vector<int> all_glasses;
    // landmark ground thruth**********************************.
    vector<int> landmark_shape(2);
    landmark_shape[0] = 1;
    landmark_shape[1] =  batch_size_ *10;
    landmark_pred_.Reshape(landmark_shape);
    landmark_gt_.Reshape(landmark_shape);
    Dtype* landmark_gt_data = landmark_gt_.mutable_cpu_data();

    // angle ground thruth**********************************.
    vector<int> angle_shape(2);
    angle_shape[0] = 1;
    angle_shape[1] =  batch_size_ *3;
    angle_pred_.Reshape(angle_shape);
    angle_gt_.Reshape(angle_shape);
    Dtype* angle_gt_data = angle_gt_.mutable_cpu_data();

    for(int item_id = 0; item_id < batch_size_; item_id++){
        int idxg = item_id*16;
        int id = label_data[idxg];
        if(id == -1)
        {
            LOG(WARNING)<<"the item_id from each image, should not be the -1!!!1";
            continue;
        }
        AnnoFaceLandmarks facemark;
        AnnoFaceOritation faceangle;

        landmark_gt_data[item_id * 10] = label_data[idxg+1]     ;
        landmark_gt_data[item_id * 10+1] = label_data[idxg+2]   ;
        landmark_gt_data[item_id * 10+2] = label_data[idxg+3]   ;
        landmark_gt_data[item_id * 10+3] = label_data[idxg+4]   ;
        landmark_gt_data[item_id * 10+4] = label_data[idxg+5]   ;
        landmark_gt_data[item_id * 10+5] = label_data[idxg+6]   ;
        landmark_gt_data[item_id * 10+6] = label_data[idxg+7]   ;
        landmark_gt_data[item_id * 10+7] = label_data[idxg+8]   ;
        landmark_gt_data[item_id * 10+8] = label_data[idxg+9]   ;
        landmark_gt_data[item_id * 10+9] = label_data[idxg+10]  ;

        angle_gt_data[item_id * 3 + 0] = label_data[idxg+11];
        angle_gt_data[item_id * 3 + 1] = label_data[idxg+12];
        angle_gt_data[item_id * 3 + 2] = label_data[idxg+13];

        all_gender.push_back(label_data[idxg+14]);
        all_glasses.push_back(label_data[idxg+15]);
    }
    CHECK_EQ(batch_size_, all_gender.size())<<"ground truth label size should match batch_size_";
    /***********************************************************************************/
    #if 0
    const Dtype* landmark_pred_data = landmark_pred_.cpu_data();
    for(int ii = 0; ii< 3; ii++)
    {
        LOG(INFO)<<"batch_index: "<<ii;
        LOG(INFO)<<"  groundtruth: "<<landmark_gt_data[ii*10]<<" "<<landmark_gt_data[ii*10+1]<<" "
                 << landmark_gt_data[ii*10+2]<<" "<< landmark_gt_data[ii*10+3]<<" "
                 << landmark_gt_data[ii*10+4]<<" "<< landmark_gt_data[ii*10+5]<<" "
                 << landmark_gt_data[ii*10+6]<<" "<< landmark_gt_data[ii*10+7]<<" "
                 << landmark_gt_data[ii*10+8]<<" "<< landmark_gt_data[ii*10+9];
        LOG(INFO)<<"  facepredata: "<<landmark_pred_data[ii*10]<<" "<<landmark_pred_data[ii*10+1]<<" "
                 << landmark_pred_data[ii*10+2]<<" "<< landmark_pred_data[ii*10+3]<<" "
                 << landmark_pred_data[ii*10+4]<<" "<< landmark_pred_data[ii*10+5]<<" "
                 << landmark_pred_data[ii*10+6]<<" "<< landmark_pred_data[ii*10+7]<<" "
                 << landmark_pred_data[ii*10+8]<<" "<< landmark_pred_data[ii*10+9];
    }
    #endif
    Dtype* landmark_pred_data = landmark_pred_.mutable_cpu_data();
    const Dtype* pred_data = bottom[0]->cpu_data();
    caffe_copy(batch_size_ *10, pred_data, landmark_pred_data);

    landmark_loss_layer_->Reshape(landmark_bottom_vec_, landmark_top_vec_);
    landmark_loss_layer_->Forward(landmark_bottom_vec_, landmark_top_vec_);

    /********************************************************************************/
    Dtype* angle_pred_data = angle_pred_.mutable_cpu_data();
    const Dtype* _pred_data = bottom[1]->cpu_data();
    caffe_copy(batch_size_ *3, _pred_data, angle_pred_data);

    angle_loss_layer_->Reshape(angle_bottom_vec_, angle_top_vec_);
    angle_loss_layer_->Forward(angle_bottom_vec_, angle_top_vec_);
    /********************************************************************************/

    // gender_loss_layer *********************************
    vector<int> gender_shape;
    if (gender_loss_type_ == MultiFaceAttriLossParameter_AttriLossType_SOFTMAX) {
        gender_shape.push_back(batch_size_);
        gender_gt_.Reshape(gender_shape);
        gender_shape.push_back(num_gender_);
        gender_pred_.Reshape(gender_shape);
    } else if (gender_loss_type_ == MultiFaceAttriLossParameter_AttriLossType_LOGISTIC) {
        gender_shape.push_back(batch_size_);
        gender_shape.push_back(num_gender_);
        gender_gt_.Reshape(gender_shape);
        gender_pred_.Reshape(gender_shape);        
    } else {
        LOG(FATAL) << "Unknown gender confidence loss type.";
    }
    Dtype* gender_gt_data = gender_gt_.mutable_cpu_data();
    caffe_set(gender_gt_.count(), Dtype(0), gender_gt_data);
    for(int ii = 0; ii < batch_size_; ii++){
        gender_gt_data[ii] = all_gender[ii];
    }
    Dtype* gender_pred_data = gender_pred_.mutable_cpu_data();
    const Dtype* gender_data = bottom[2]->cpu_data();
    caffe_copy(batch_size_ *num_gender_, gender_data, gender_pred_data);
    
    gender_loss_layer_->Reshape(gender_bottom_vec_, gender_top_vec_);
    gender_loss_layer_->Forward(gender_bottom_vec_, gender_top_vec_);

    /********************************************************************************/
    // glasses_loss_layer_********************************
    vector<int> glasses_shape;
    if (glasses_loss_type_ == MultiFaceAttriLossParameter_AttriLossType_SOFTMAX) {
        glasses_shape.push_back(batch_size_);
        glasses_gt_.Reshape(glasses_shape);
        glasses_shape.push_back(num_glasses_);
        glasses_pred_.Reshape(glasses_shape);
    } else if (glasses_loss_type_ == MultiFaceAttriLossParameter_AttriLossType_LOGISTIC) {
        glasses_shape.push_back(batch_size_);
        glasses_shape.push_back(num_glasses_);
        glasses_gt_.Reshape(glasses_shape);
        glasses_pred_.Reshape(glasses_shape);
    } else {
        LOG(FATAL) << "Unknown glasses confidence loss type.";
    }

    Dtype* glasses_pred_data = glasses_pred_.mutable_cpu_data();
    const Dtype* glasses_data = bottom[3]->cpu_data();
    caffe_copy(batch_size_ *num_glasses_, glasses_data, glasses_pred_data);
    Dtype* glasses_gt_data = glasses_gt_.mutable_cpu_data();
    caffe_set(glasses_gt_.count(), Dtype(0), glasses_gt_data);
    for(int ii = 0; ii < batch_size_; ii++){
        glasses_gt_data[ii] = all_glasses[ii];
    }
    
    glasses_loss_layer_->Reshape(glasses_bottom_vec_, glasses_top_vec_);
    glasses_loss_layer_->Forward(glasses_bottom_vec_, glasses_top_vec_);

    /**************************************sum loss value****************************/
    top[0]->mutable_cpu_data()[0] = 0;
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, batch_size_, 1, -1);
    if (this->layer_param_.propagate_down(0)) {
    top[0]->mutable_cpu_data()[0] +=
        landmark_weight_ * landmark_loss_.cpu_data()[0] / normalizer;
    }
    if (this->layer_param_.propagate_down(1)) {
    top[0]->mutable_cpu_data()[0] += 
            angle_weight_*angle_loss_.cpu_data()[0] / normalizer;
    }
    if (this->layer_param_.propagate_down(2)) {
    top[0]->mutable_cpu_data()[0] += 
            gender_weight_*gender_loss_.cpu_data()[0] / normalizer;
    }
    if(this->layer_param_.propagate_down(3)) {
    top[0]->mutable_cpu_data()[0] += 
            glasses_weight_*glasses_loss_.cpu_data()[0] / normalizer;
    }
    #if 0
    LOG(INFO)<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
    LOG(INFO)<<"total origin facepoint_loss_: "<< landmark_loss_.cpu_data()[0];
    LOG(INFO)<<"total origin _gender_loss_: "<< gender_loss_.cpu_data()[0];
    LOG(INFO)<<"total origin glassess_loss_: "<< glasses_loss_.cpu_data()[0];
    LOG(INFO)<<"total loss_layer loss value: "<<top[0]->cpu_data()[0]
             <<" normalizer: "<<normalizer;
    //LOG(FATAL)<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
    #endif
}

template <typename Dtype>
void MultiFaceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down,
const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[4]) {
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
        // Copy gradient back to bottom[0].
        /*const Dtype* landmark_pred_diff = landmark_pred_.cpu_diff();
        for (int ii = 0; ii < batch_size_; ++ii) {
            caffe_copy<Dtype>(10, landmark_pred_diff + ii * 10,
                                mark_bottom_diff + ii*10);
            mark_bottom_diff += bottom[0]->offset(1);
        }*/
    }
    if (propagate_down[1]) {
        Dtype* angle_bottom_diff = bottom[1]->mutable_cpu_diff();
        caffe_set(bottom[1]->count(), Dtype(0), angle_bottom_diff);
        vector<bool> angle_propagate_down;
        // Only back propagate on prediction, not ground truth.
        angle_propagate_down.push_back(true);
        angle_propagate_down.push_back(false);
        angle_loss_layer_->Backward(angle_top_vec_, angle_propagate_down,
                                angle_bottom_vec_);
        // Scale gradient.
        Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
            normalization_, batch_size_, 1, -1);
        Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
        
        caffe_scal(angle_pred_.count(), loss_weight, angle_pred_.mutable_cpu_diff());
        bottom[1]->ShareDiff(angle_pred_);
    }

    /*************************************************************************************/
    // Back propagate on gender confidence prediction.
    if (propagate_down[2]) {
        Dtype* gender_bottom_diff = bottom[2]->mutable_cpu_diff();
        caffe_set(bottom[2]->count(), Dtype(0), gender_bottom_diff);
        vector<bool> gender_propagate_down;
        // Only back propagate on prediction, not ground truth.
        gender_propagate_down.push_back(true);
        gender_propagate_down.push_back(false);
        gender_loss_layer_->Backward(gender_top_vec_, gender_propagate_down,
                                    gender_bottom_vec_);
        // Scale gradient.
        Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
            normalization_, batch_size_, 1, -1);
        Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
        caffe_scal(gender_pred_.count(), loss_weight,
                    gender_pred_.mutable_cpu_diff());
        // Copy gradient back to bottom[1].
        // The diff is already computed and stored.
        bottom[2]->ShareDiff(gender_pred_);
    }

    /*************************************************************************************/
    // Back propagate on glasses confidence prediction.
    if (propagate_down[3]) {
        Dtype* glasses_bottom_diff = bottom[3]->mutable_cpu_diff();
        caffe_set(bottom[3]->count(), Dtype(0), glasses_bottom_diff);
        vector<bool> glasses_propagate_down;
        // Only back propagate on prediction, not ground truth.
        glasses_propagate_down.push_back(true);
        glasses_propagate_down.push_back(false);
        glasses_loss_layer_->Backward(glasses_top_vec_, glasses_propagate_down,
                                    glasses_bottom_vec_);
        // Scale gradient.
        Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
            normalization_, batch_size_, 1, -1);
        Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
        caffe_scal(glasses_pred_.count(), loss_weight,
                    glasses_pred_.mutable_cpu_diff());
        bottom[3]->ShareDiff(glasses_pred_);
    }
}

INSTANTIATE_CLASS(MultiFaceLossLayer);
REGISTER_LAYER_CLASS(MultiFaceLoss);

}  // namespace caffe
