#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/cosin_large_margin_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {
    template <typename Dtype>
    void CosinLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        const int num_output = this->layer_param_.cosin_loss_param().num_output();
        margin_ = this->layer_param_.cosin_loss_param().margin();
        scaler_ = this->layer_param_.cosin_loss_param().scale();
        Num_Class_ = num_output;
        Num_BatchSize_ = bottom[0]->num();
        const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.cosin_loss_param().axis());
        feature_Dim_ = bottom[0]->count(axis);
        if (this->blobs_.size() > 0) {
            LOG(INFO) << "Skipping parameter initialization";
        } else {
            this->blobs_.resize(1);
            // Intialize the weight
            vector<int> cosin_shape(2);
            cosin_shape[0] = Num_Class_;
            cosin_shape[1] = feature_Dim_;
            this->blobs_[0].reset(new Blob<Dtype>(cosin_shape));
            // fill the weights
            shared_ptr<Filler<Dtype> > cosin_filler(GetFiller<Dtype>(
                this->layer_param_.cosin_loss_param().cosin_filler()));
            cosin_filler->Fill(this->blobs_[0].get());
        }  // 参数初始化
        this->param_propagate_down_.resize(this->blobs_.size(), true);
        Normalise_Weight_.ReshapeLike(*(this->blobs_[0]));
        Normalise_feature_.ReshapeLike(*(bottom[0]));
        vector<int> normail_shape(2);
        normail_shape[0] = Num_BatchSize_;
        normail_shape[1] = Num_Class_;
        Normalizer_cosValue_.reset(new Blob<Dtype>(normail_shape));
        /******************softmax entropy loss **************************/
        conf_bottom_vec_.push_back(Normalizer_cosValue_);
        conf_bottom_vec_.push_back(conf_gt_);
        vector<int> loss_shape(1, 1);
        conf_loss_.Reshape(loss_shape);
        conf_top_vec_.push_back(&conf_loss_);
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
        layer_param.set_type("SoftmaxWithLoss");
        layer_param.add_loss_weight(Dtype(1.));
        layer_param.mutable_loss_param()->set_normalization(
            LossParameter_NormalizationMode_NONE);
        SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
        softmax_param->set_axis(1);
        // Fake reshape.
        vector<int> conf_shape(1, 1);
        conf_gt_.Reshape(conf_shape);
        conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
    }


    template <typename Dtype>
    void CosinLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
        CHECK_EQ(bottom[1]->channels(), 1);
        CHECK_EQ(bottom[1]->height(), 1);
        CHECK_EQ(bottom[1]->width(), 1);

        LossLayer<Dtype>::Reshape(bottom, top);


    }
    template <typename Dtype>
    void CosinLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        const Dtype* label = bottom[1]->cpu_data();

        Normalise_Weight_ = vector_L2_Normalise(&(this->blobs_[0]));
        Normalise_feature_ = vector_L2_Normalise(bottom[0]);
        const Dtype * weight = Normalise_Weight_.cpu_data();
        const Dtype * feature = Normalise_feature_.cpu_data();
        Dtype * wx_cos_value = Normalizer_cosValue_.mutable_cpu_data();
        caffe_cpu_gemm(CblasNoTrans, CblasTrans, Num_BatchSize_, Num_Class_, feature_Dim_, Dtype(1.0)
                        feature, weight, Dtype(0.0), wx_cos_value);
        for(int i = 0; i< Num_BatchSize_, i ++){
            caffe_add_scalar(Num_Class_, margin_, wx_cos_value + i *Num_Class_);
            caffe_scal(Num_Class_, scaler_, wx_cos_value + i *Num_Class_);
        }
        vector<int> conf_shape;
        conf_shape.push_back(Num_BatchSize_);
        conf_gt_.Reshape(conf_shape);
        Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
        for(int i= 0; i < Num_BatchSize_; i++)
            conf_gt_data[i] = label[i];
        conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
        conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);
        top[0]->mutable_cpu_data()[0] = 0;
        top[0]->mutable_cpu_data()[0] += conf_loss_.cpu_data()[0] / Num_BatchSize_;
        #if 1
        LOG(INFO)<<"cosin face LMCL loss value: " << top[0]->mutable_cpu_data()[0];
        #endif
    }
    template <typename Dtype>
    void CosinLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        ;
        if (propagate_down[1]) {
            LOG(FATAL) << this->type()<< " Layer cannot backpropagate to label inputs.";
  }
    }
    #ifdef CPU_ONLY
    STUB_GPU(CosinLossLayer);
    #endif

    INSTANTIATE_CLASS(CosinLossLayer);
    REGISTER_LAYER_CLASS(CosinLoss);
}//name caffe