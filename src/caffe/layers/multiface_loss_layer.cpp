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
    const MultiFaceLossParameter& multiface_loss_param =
        this->layer_param_.multiface_loss_param();
    multiface_loss_param_ = this->layer_param_.multiface_loss_param();

    batch_size_ = bottom[0]->num();
    // Get other parameters.
    CHECK_EQ(multiface_loss_param.num_gender(), 2) << "Must provide num_gender, and the num_gender must is 2.";
    CHECK_EQ(multiface_loss_param.num_glasses(), 2) << "Must prodived num_glasses, and the num_glasses must is 2";
    CHECK_EQ(multiface_loss_param.num_headpose(), 5) << "Must provide num_headpose, and the num_headpose must is 5";
    num_gender_ = multiface_loss_param.num_gender();
    num_glasses_ = multiface_loss_param.num_glasses();
    num_headpose_ = multiface_loss_param.num_headpose();
    //share_location_ = multiface_loss_param.share_location();
    //loc_classes_ = share_location_ ? 1 : num_classes_;
    //background_label_id_ = multiface_loss_param.background_label_id();

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

    /***************************************************************************************/
    // Set up headpose confidence loss layer.
    headpose_loss_type_ = multiface_loss_param.conf_headpose_loss_type();
    headpose_bottom_vec_.push_back(&headpose_pred_);
    headpose_bottom_vec_.push_back(&headpose_gt_);
    headpose_loss_.Reshape(loss_shape);
    headpose_top_vec_.push_back(&headpose_loss_);
    if (headpose_loss_type_ == MultiFaceLossParameter_AttriLossType_SOFTMAX) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_softmax_headpose_conf");
        layer_param.set_type("SoftmaxWithLoss");
        layer_param.add_loss_weight(Dtype(1.));
        layer_param.mutable_loss_param()->set_normalization(
            LossParameter_NormalizationMode_NONE);
        SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
        softmax_param->set_axis(1);
        // Fake reshape.
        vector<int> headpose_shape(1, 1);
        headpose_gt_.Reshape(headpose_shape);
        headpose_shape.push_back(num_headpose_);
        headpose_pred_.Reshape(headpose_shape);
        headpose_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        headpose_loss_layer_->SetUp(headpose_bottom_vec_, headpose_top_vec_);
    } else if (headpose_loss_type_ == MultiFaceLossParameter_AttriLossType_LOGISTIC) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_logistic_headpose_conf");
        layer_param.set_type("SigmoidCrossEntropyLoss");
        layer_param.add_loss_weight(Dtype(1.));
        // Fake reshape.
        vector<int> headpose_shape(1, 1);
        headpose_shape.push_back(num_headpose_);
        headpose_gt_.Reshape(headpose_shape);
        headpose_pred_.Reshape(headpose_shape);
        headpose_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        headpose_loss_layer_->SetUp(headpose_bottom_vec_, headpose_top_vec_);
    } else {
        LOG(FATAL) << "Unknown face attributes headpose loss type.";
    }
}

template <typename Dtype>
void MultiFaceLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    int num_landmarks = bottom[0]->shape(1);  //bottom[0]: landmarks , 
    int num_gender = bottom[1]->shape(1); //bottom[1]: num_gender;
    int num_glasses = bottom[2]->shape(1); // bottom[2]: num_glasses;
    int num_headpose = bottom[3]->shape(1); //bottom[3]: num_headpose
    CHECK_EQ(num_landmarks, 10)<<"number of lanmarks point value must equal to 10";
    CHECK_EQ(num_gender_, num_gender)<<"number of gender must match prototxt provided";
    CHECK_EQ(num_glasses_, num_glasses)<<"number of glasses must match prototxt provided";
    CHECK_EQ(num_headpose_, num_headpose)<<"number of headpose must match prototxt provided";
}

template <typename Dtype>
void MultiFaceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
    const Dtype* label_data = bottom[4]->cpu_data();
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
    map<int, LandmarkFace > all_landmarks;
    vector<int> all_gender;
    vector<int> all_glasses;
    vector<int> all_headpose; 
    all_landmarks.clear();
    for(int item_id = 0; item_id < batch_size_; item_id++){
        int idx = item_id*14;
        int id = label_data[idx];
        if(id == -1)
        {
            LOG(WARNING)<<"the item_id from each image, should not be the -1!!!1";
            continue;
        }
        LandmarkFace facemark;
        facemark.set_x1(label_data[idx+1]);
        facemark.set_x2(label_data[idx+2]);
        facemark.set_x3(label_data[idx+3]);
        facemark.set_x4(label_data[idx+4]);
        facemark.set_x5(label_data[idx+5]);
        facemark.set_y1(label_data[idx+6]);
        facemark.set_y2(label_data[idx+7]);
        facemark.set_y3(label_data[idx+8]);
        facemark.set_y4(label_data[idx+9]);
        facemark.set_y5(label_data[idx+10]);
        all_gender.push_back(label_data[idx+11]);
        all_glasses.push_back(label_data[idx+12]);
        all_headpose.push_back(label_data[idx+13]);
        all_landmarks.insert(pair<int,LandmarkFace>(item_id, facemark));
    }
    CHECK_EQ(batch_size_, all_landmarks.size())<<"ground truth label size should match batch_size_";
    /***********************************************************************************/
    // Form data to pass on to landmark_loss_layer_.
    vector<int> landmark_shape(2);
    landmark_shape[0] = 1;
    landmark_shape[1] = batch_size_ * 10;
    landmark_pred_.Reshape(landmark_shape);
    landmark_gt_.Reshape(landmark_shape);
    Blob<Dtype> landmark_temp;
    landmark_temp.ReshapeLike(*(bottom[0]));
    landmark_temp.CopyFrom(*(bottom[0]));
    landmark_temp.Reshape(landmark_shape);
    landmark_pred_.CopyFrom(landmark_temp);
    Dtype* landmark_gt_data = landmark_gt_.mutable_cpu_data();
    int count =0;
    for(int ii = 0; ii< batch_size_; ii++)
    {
        LandmarkFace face = all_landmarks[ii];
        landmark_gt_data[count*10] = face.x1() ;
        landmark_gt_data[count*10+1] = face.x2() ;
        landmark_gt_data[count*10+2] = face.x3() ;
        landmark_gt_data[count*10+3] = face.x4() ;
        landmark_gt_data[count*10+4] = face.x5() ;
        landmark_gt_data[count*10+5] = face.y1() ;
        landmark_gt_data[count*10+6] = face.y2() ;
        landmark_gt_data[count*10+7] = face.y3() ;
        landmark_gt_data[count*10+8] = face.y4() ;
        landmark_gt_data[count*10+9] = face.y5() ;
        
        ++count;
    }
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
        gender_shape.push_back(1);
        gender_shape.push_back(batch_size_);
        gender_shape.push_back(num_gender_);
        gender_gt_.Reshape(gender_shape);
        gender_pred_.Reshape(gender_shape);
    } else {
        LOG(FATAL) << "Unknown gender confidence loss type.";
    }
    //Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
    gender_pred_.CopyFrom(*(bottom[1]));
    Dtype* gender_gt_data = gender_gt_.mutable_cpu_data();
    caffe_set(gender_gt_.count(), Dtype(0), gender_gt_data);
    for(int ii = 0; ii< batch_size_; ii++)
    {
        gender_gt_data[ii] = all_gender[ii];
    }
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
        glasses_shape.push_back(1);
        glasses_shape.push_back(batch_size_);
        glasses_shape.push_back(num_glasses_);
        glasses_gt_.Reshape(glasses_shape);
        glasses_pred_.Reshape(glasses_shape);
    } else {
        LOG(FATAL) << "Unknown glasses confidence loss type.";
    }
    //Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
    glasses_pred_.CopyFrom(*(bottom[2]));
    Dtype* glasses_gt_data = glasses_gt_.mutable_cpu_data();
    caffe_set(glasses_gt_.count(), Dtype(0), glasses_gt_data);
    for(int ii = 0; ii< batch_size_; ii++)
    {
        glasses_gt_data[ii] = all_glasses[ii];
    }
    glasses_loss_layer_->Reshape(glasses_bottom_vec_, glasses_top_vec_);
    glasses_loss_layer_->Forward(glasses_bottom_vec_, glasses_top_vec_);

    /********************************************************************************/
    // Form data to pass on to headpose_loss_layer_.
    vector<int> headpose_shape;
    if (headpose_loss_type_ == MultiFaceLossParameter_AttriLossType_SOFTMAX) {
        headpose_shape.push_back(batch_size_);
        headpose_gt_.Reshape(headpose_shape);
        headpose_shape.push_back(num_headpose_);
        headpose_pred_.Reshape(headpose_shape);
    } else if (headpose_loss_type_ == MultiFaceLossParameter_AttriLossType_LOGISTIC) {
        headpose_shape.push_back(1);
        headpose_shape.push_back(batch_size_);
        headpose_shape.push_back(num_headpose_);
        headpose_gt_.Reshape(headpose_shape);
        headpose_pred_.Reshape(headpose_shape);
    } else {
        LOG(FATAL) << "Unknown headpose confidence loss type.";
    }
    //Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
    headpose_pred_.CopyFrom(*(bottom[3]));
    Dtype* headpose_gt_data = headpose_gt_.mutable_cpu_data();
    caffe_set(headpose_gt_.count(), Dtype(0), headpose_gt_data);
    for(int ii = 0; ii< batch_size_; ii++)
    {
        headpose_gt_data[ii] = all_headpose[ii];
    }
    headpose_loss_layer_->Reshape(headpose_bottom_vec_, headpose_top_vec_);
    headpose_loss_layer_->Forward(headpose_bottom_vec_, headpose_top_vec_);

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
            gender_loss_.cpu_data()[0] / normalizer;
    }
    if(this->layer_param_.propagate_down(2)) {
    top[0]->mutable_cpu_data()[0] += 
            glasses_loss_.cpu_data()[0] / normalizer;
    }
    if(this->layer_param_.propagate_down(3)) {
    top[0]->mutable_cpu_data()[0] += 
            headpose_loss_.cpu_data()[0] / normalizer;
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
        // Copy gradient back to bottom[0].
        const Dtype* landmark_pred_diff = landmark_pred_.cpu_diff();
        for (int ii = 0; ii < batch_size_; ++ii) {
            caffe_copy<Dtype>(10, landmark_pred_diff + ii * 10,
                                mark_bottom_diff + ii*10);
            mark_bottom_diff += bottom[0]->offset(1);
        }
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

    /*************************************************************************************/
    // Back propagate on headpose confidence prediction.
if (propagate_down[3]) {
        Dtype* headpose_bottom_diff = bottom[3]->mutable_cpu_diff();
        caffe_set(bottom[3]->count(), Dtype(0), headpose_bottom_diff);
        vector<bool> headpose_propagate_down;
        // Only back propagate on prediction, not ground truth.
        headpose_propagate_down.push_back(true);
        headpose_propagate_down.push_back(false);
        headpose_loss_layer_->Backward(headpose_top_vec_, headpose_propagate_down,
                                    headpose_bottom_vec_);
        // Scale gradient.
        Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
            normalization_, batch_size_, 1, -1);
        Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
        caffe_scal(headpose_pred_.count(), loss_weight,
                    headpose_pred_.mutable_cpu_diff());
        // Copy gradient back to bottom[1].
        // The diff is already computed and stored.
        bottom[3]->ShareDiff(headpose_pred_);
    }
}

INSTANTIATE_CLASS(MultiFaceLossLayer);
REGISTER_LAYER_CLASS(MultiFaceLoss);

}  // namespace caffe
