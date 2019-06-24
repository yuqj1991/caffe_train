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
        this->layer_param_.add_propagate_down(false);
    }
    const MultiFaceLossParameter& multiface_loss_param =
        this->layer_param_.multiface_loss_param();
    multiface_loss_param_ = this->layer_param_.multiface_loss_param();

    batch_size_ = bottom[0]->num();
    // Get other parameters.
    CHECK_EQ(multiface_loss_param.num_gender(), 2) << "Must provide num_gender, and the num_gender must is 2.";
    CHECK_EQ(multiface_loss_param.num_glasses(), 2) << "Must prodived num_glasses, and the num_glasses must is 2";
    num_gender_ = multiface_loss_param.num_gender();
    num_glasses_ = multiface_loss_param.num_glasses();
    
    /*
    mining_type_ = multiface_loss_param.mining_type();
    if (multiface_loss_param.has_do_neg_mining()) {
    LOG(WARNING) << "do_neg_mining is deprecated, use mining_type instead.";
    do_neg_mining_ = multiface_loss_param.do_neg_mining();
    CHECK_EQ(do_neg_mining_,
                mining_type_ != MultiBoxLossParameter_MiningType_NONE);
    }
    do_neg_mining_ = mining_type_ != MultiBoxLossParameter_MiningType_NONE;
    */
    if (!this->layer_param_.loss_param().has_normalization() &&
        this->layer_param_.loss_param().has_normalize()) {
        normalization_ = this->layer_param_.loss_param().normalize() ?
                        LossParameter_NormalizationMode_VALID :
                        LossParameter_NormalizationMode_BATCH_SIZE;
    } else {
        normalization_ = this->layer_param_.loss_param().normalization();
    }
    #if 0
    if (do_neg_mining_) {
    CHECK(share_location_)
        << "Currently only support negative mining if share_location is true.";
    }
    #endif

    vector<int> loss_shape(1, 1);
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
    if (landmark_loss_type_ == MultiFaceLossParameter_MarkLossType_L2) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_l2_loc");
        layer_param.set_type("EuclideanLoss");
        layer_param.add_loss_weight(landmark_weight_);
        landmark_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        landmark_loss_layer_->SetUp(landmark_bottom_vec_, landmark_top_vec_);
    } else if (landmark_loss_type_ == MultiFaceLossParameter_MarkLossType_SMOOTH_L1) {
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
    // Set up gender confidence loss layer.
    gender_weight_ = multiface_loss_param.gender_weight();
    gender_loss_type_ = multiface_loss_param.conf_gender_loss_type();
    gender_bottom_vec_.push_back(&gender_pred_);
    gender_bottom_vec_.push_back(&gender_gt_);
    gender_loss_.Reshape(loss_shape);
    gender_top_vec_.push_back(&gender_loss_);
    if (gender_loss_type_ == MultiFaceLossParameter_AttriLossType_SOFTMAX) {
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
    } else if (gender_loss_type_ == MultiFaceLossParameter_AttriLossType_LOGISTIC) {
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
    if (glasses_loss_type_ == MultiFaceLossParameter_AttriLossType_SOFTMAX) {
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
    } else if (glasses_loss_type_ == MultiFaceLossParameter_AttriLossType_LOGISTIC) {
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
    int num_gender = bottom[1]->shape(1); //bottom[1]: num_gender;
    int num_glasses = bottom[2]->shape(1); // bottom[2]: num_glasses;
    CHECK_EQ(num_landmarks, 10)<<"number of lanmarks point value must equal to 10";
    CHECK_EQ(num_gender_, num_gender)<<"number of gender must match prototxt provided";
    CHECK_EQ(num_glasses_, num_glasses)<<"number of glasses must match prototxt provided";
}

template <typename Dtype>
void MultiFaceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
    const Dtype* label_data = bottom[3]->cpu_data();
 
    /***************************************retrive all ground truth****************************************/
    // Retrieve all landmarks , gender, and glasses.
    map<int, LandmarkFace > all_landmarks;
    vector<int> all_gender;
    vector<int> all_glasses;
    all_landmarks.clear();
    #if 0
    LOG(INFO)<< "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
    const Dtype* pre_point_data = bottom[0]->cpu_data();
    for(int ii = 0; ii< batch_size_; ii++)
    {
        int idx = ii*10;
        int idxg = ii*13;
        LOG(INFO)<<"batch_index: "<<ii;
        LOG(INFO)<<" pre layers: "<< pre_point_data[idx]<<" "<< pre_point_data[idx+1]<<" "
                 << pre_point_data[idx+2]<<" "<< pre_point_data[idx+3]<<" "
                 << pre_point_data[idx+4]<<" "<< pre_point_data[idx+5]<<" "
                 << pre_point_data[idx+6]<<" "<< pre_point_data[idx+7]<<" "
                 << pre_point_data[idx+8]<<" "<< pre_point_data[idx+9];
        LOG(INFO)<<" groundtruth: "<<label_data[idxg]<<" "<<label_data[idxg+1]<<" "
                 << label_data[idxg+2]<<" "<< label_data[idxg+3]<<" "
                 << label_data[idxg+4]<<" "<< label_data[idxg+5]<<" "
                 << label_data[idxg+6]<<" "<< label_data[idxg+7]<<" "
                 << label_data[idxg+8]<<" "<< label_data[idxg+9]<<" "
                 << label_data[idxg+10]<<" "<< label_data[idxg+11]<<" "
                 << label_data[idxg+12]<<" "<< label_data[idxg+13];
    }
    #endif
    for(int item_id = 0; item_id < batch_size_; item_id++){
        int idxg = item_id*13;
        int id = label_data[idxg];
        if(id == -1)
        {
            LOG(WARNING)<<"the item_id from each image, should not be the -1!!!1";
            continue;
        }
        LandmarkFace facemark;
        facemark.set_x1(label_data[idxg+1]);
        facemark.set_x2(label_data[idxg+2]);
        facemark.set_x3(label_data[idxg+3]);
        facemark.set_x4(label_data[idxg+4]);
        facemark.set_x5(label_data[idxg+5]);
        facemark.set_y1(label_data[idxg+6]);
        facemark.set_y2(label_data[idxg+7]);
        facemark.set_y3(label_data[idxg+8]);
        facemark.set_y4(label_data[idxg+9]);
        facemark.set_y5(label_data[idxg+10]);
        all_gender.push_back(label_data[idxg+11]);
        all_glasses.push_back(label_data[idxg+12]);
        all_landmarks.insert(pair<int,LandmarkFace>(item_id, facemark));
    }
    CHECK_EQ(batch_size_, all_landmarks.size())<<"ground truth label size should match batch_size_";
    /***********************************************************************************/
    // Form data to pass on to landmark_loss_layer_.
   
    vector<int> landmark_shape(2);
    landmark_shape[0] = 1;
    landmark_shape[1] =  batch_size_ *10;
    landmark_pred_.Reshape(landmark_shape);
    landmark_gt_.Reshape(landmark_shape);
    Dtype* landmark_gt_data = landmark_gt_.mutable_cpu_data();
    for(int ii = 0; ii< batch_size_; ii++)
    {
        LandmarkFace face = all_landmarks[ii];
        landmark_gt_data[ii*10] = face.x1() ;
        landmark_gt_data[ii*10+1] = face.x2() ;
        landmark_gt_data[ii*10+2] = face.x3() ;
        landmark_gt_data[ii*10+3] = face.x4() ;
        landmark_gt_data[ii*10+4] = face.x5() ;
        landmark_gt_data[ii*10+5] = face.y1() ;
        landmark_gt_data[ii*10+6] = face.y2() ;
        landmark_gt_data[ii*10+7] = face.y3() ;
        landmark_gt_data[ii*10+8] = face.y4() ;
        landmark_gt_data[ii*10+9] = face.y5() ;
    }
    Dtype* landmark_pred_data = landmark_pred_.mutable_cpu_data();
    const Dtype* pred_data = bottom[0]->cpu_data();
    caffe_copy(batch_size_ *10, pred_data, landmark_pred_data);
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
    landmark_loss_layer_->Reshape(landmark_bottom_vec_, landmark_top_vec_);
    landmark_loss_layer_->Forward(landmark_bottom_vec_, landmark_top_vec_);

    /********************************************************************************/
    // Form data to pass on to gender_loss_layer_.
    // Reshape the gender confidence data.
    vector<int> gender_shape;
    if (gender_loss_type_ == MultiFaceLossParameter_AttriLossType_SOFTMAX) {
        gender_shape.push_back(batch_size_);
        gender_gt_.Reshape(gender_shape);
        gender_shape.push_back(num_gender_);
        gender_pred_.Reshape(gender_shape);
    } else if (gender_loss_type_ == MultiFaceLossParameter_AttriLossType_LOGISTIC) {
        gender_shape.push_back(batch_size_);
        gender_shape.push_back(num_gender_);
        gender_gt_.Reshape(gender_shape);
        gender_pred_.Reshape(gender_shape);
        /************************************************/
        
    } else {
        LOG(FATAL) << "Unknown gender confidence loss type.";
    }
    //Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
    Dtype* gender_gt_data = gender_gt_.mutable_cpu_data();
    caffe_set(gender_gt_.count(), Dtype(0), gender_gt_data);
    for(int ii = 0; ii< batch_size_; ii++)
    {
        gender_gt_data[ii] = all_gender[ii];
    }
    Dtype* gender_pred_data = gender_pred_.mutable_cpu_data();
    const Dtype* gender_data = bottom[1]->cpu_data();
    caffe_copy(batch_size_ *num_gender_, gender_data, gender_pred_data);
    #if 0
    const Dtype* gender_pred_data = gender_pred_.cpu_data();
    const Dtype* bottom_pred_data = bottom[1]->cpu_data();
    for(int ii = 0; ii< 3; ii++)
    {
        LOG(INFO)<< "gender_gt_data: "<<gender_gt_data[ii];
        LOG(INFO)<< "gender_pr_data: "<<gender_pred_data[ii*2]<<" "<<gender_pred_data[ii*2+1];
        LOG(INFO)<< "bottom_01_data: "<<bottom_pred_data[ii*2]<<" "<<bottom_pred_data[ii*2+1];
    }
    #endif
    gender_loss_layer_->Reshape(gender_bottom_vec_, gender_top_vec_);
    gender_loss_layer_->Forward(gender_bottom_vec_, gender_top_vec_);

    /********************************************************************************/
    // Form data to pass on to glasses_loss_layer_.
    vector<int> glasses_shape;
    if (glasses_loss_type_ == MultiFaceLossParameter_AttriLossType_SOFTMAX) {
        glasses_shape.push_back(batch_size_);
        glasses_gt_.Reshape(glasses_shape);
        glasses_shape.push_back(num_glasses_);
        glasses_pred_.Reshape(glasses_shape);
    } else if (glasses_loss_type_ == MultiFaceLossParameter_AttriLossType_LOGISTIC) {
        glasses_shape.push_back(batch_size_);
        glasses_shape.push_back(num_glasses_);
        glasses_gt_.Reshape(glasses_shape);
        glasses_pred_.Reshape(glasses_shape);
    } else {
        LOG(FATAL) << "Unknown glasses confidence loss type.";
    }
    //Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
    Dtype* glasses_pred_data = glasses_pred_.mutable_cpu_data();
    const Dtype* glasses_data = bottom[2]->cpu_data();
    caffe_copy(batch_size_ *num_glasses_, glasses_data, glasses_pred_data);

    Dtype* glasses_gt_data = glasses_gt_.mutable_cpu_data();
    caffe_set(glasses_gt_.count(), Dtype(0), glasses_gt_data);
    for(int ii = 0; ii< batch_size_; ii++)
    {
        glasses_gt_data[ii] = all_glasses[ii];
    }
    #if 0
    const Dtype* glasses_pred_data = glasses_pred_.cpu_data();
    const Dtype* bottom2_pred_data = bottom[2]->cpu_data();
    for(int ii = 0; ii< 3; ii++)
    {
        LOG(INFO)<< "glasses_gt_data: "<<glasses_gt_data[ii];
        LOG(INFO)<< "glasses_pr_data: "<<glasses_pred_data[ii*2]<<" "<<glasses_pred_data[ii*2+1];
        LOG(INFO)<< "bottom_02_data: "<<bottom2_pred_data[ii*2]<<" "<<bottom2_pred_data[ii*2+1];
    }
    #endif
    glasses_loss_layer_->Reshape(glasses_bottom_vec_, glasses_top_vec_);
    glasses_loss_layer_->Forward(glasses_bottom_vec_, glasses_top_vec_);

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
            gender_weight_*gender_loss_.cpu_data()[0] / normalizer;
    }
    if(this->layer_param_.propagate_down(2)) {
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
    if (propagate_down[3]) {
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
        #if 0
            LOG(INFO)<<"top[0]->cpu_diff()[0]: "<<top[0]->cpu_diff()[0];
        #endif
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

    /*************************************************************************************/
    // Back propagate on gender confidence prediction.
    if (propagate_down[1]) {
        Dtype* gender_bottom_diff = bottom[1]->mutable_cpu_diff();
        caffe_set(bottom[1]->count(), Dtype(0), gender_bottom_diff);
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
        bottom[1]->ShareDiff(gender_pred_);
    }

    /*************************************************************************************/
    // Back propagate on glasses confidence prediction.
    if (propagate_down[2]) {
        Dtype* glasses_bottom_diff = bottom[2]->mutable_cpu_diff();
        caffe_set(bottom[2]->count(), Dtype(0), glasses_bottom_diff);
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
        // Copy gradient back to bottom[2].
        // The diff is already computed and stored.
        bottom[2]->ShareDiff(glasses_pred_);
    }
}

INSTANTIATE_CLASS(MultiFaceLossLayer);
REGISTER_LAYER_CLASS(MultiFaceLoss);

}  // namespace caffe
