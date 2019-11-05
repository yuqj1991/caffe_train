#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/cosin_l2_normalize_innerproduct.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {
    template <typename Dtype>
    void CosinL2NormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        if (this->layer_param_.propagate_down_size() == 0) {
            this->layer_param_.add_propagate_down(true);
            this->layer_param_.add_propagate_down(false);
        }
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
            vector<int> cosin_shape(2);
            cosin_shape[0] = Num_Class_;
            cosin_shape[1] = feature_Dim_;
            this->blobs_[0].reset(new Blob<Dtype>(cosin_shape));
            shared_ptr<Filler<Dtype> > cosin_filler(GetFiller<Dtype>(
                this->layer_param_.cosin_loss_param().cosin_filler()));
            cosin_filler->Fill(this->blobs_[0].get());
        } 
        this->param_propagate_down_.resize(this->blobs_.size(), true);
        Normalise_Weight_.ReshapeLike(*(this->blobs_[0]));
        Normalise_feature_.ReshapeLike(*(bottom[0]));
    }

    template <typename Dtype>
    void CosinL2NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
        vector<int> normail_shape(2);
        normail_shape[0] = Num_BatchSize_;
        normail_shape[1] = Num_Class_;
        top[0]->Reshape(normail_shape);
        /*****************************/
        normail_shape[0] = Num_Class_;
        normail_shape[1] = feature_Dim_;
        Normalise_Weight_.Reshape(normail_shape);
        /*****************************/
        normail_shape[0] = Num_BatchSize_;
        normail_shape[1] = feature_Dim_;
        Normalise_feature_.Reshape(normail_shape);
    }

    template <typename Dtype>
    void CosinL2NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        const Dtype* raw_weight = this->blobs_[0]->cpu_data();
        const Dtype* raw_feature = bottom[0]->cpu_data();
        const Dtype* label_data = bottom[1]->cpu_data();
        Dtype * normalize_weight_data = Normalise_Weight_.mutable_cpu_data();
        Dtype * normalize_feature_data = Normalise_feature_.mutable_cpu_data();
        vector_L2_Normalise(raw_weight, Num_Class_, feature_Dim_, normalize_weight_data);
        vector_L2_Normalise(raw_feature, Num_BatchSize_, feature_Dim_, normalize_feature_data);
        const Dtype * weight = Normalise_Weight_.cpu_data();
        const Dtype * feature = Normalise_feature_.cpu_data();
        Dtype * wx_cos_value = top[0]->mutable_cpu_data();
        caffe_cpu_gemm(CblasNoTrans, CblasTrans, Num_BatchSize_, Num_Class_, feature_Dim_, Dtype(1.0)
                        , feature, weight, Dtype(0.0), wx_cos_value);
        for(int i = 0; i< Num_BatchSize_; i ++){
            const int label_value = static_cast<int>(label_data[i]);
            caffe_add_scalar(1, (Dtype)margin_, wx_cos_value + i *Num_Class_ + label_value);
            caffe_scal(Num_Class_, (Dtype)scaler_, wx_cos_value + i *Num_Class_);
        }
    }

    template <typename Dtype>
    void CosinL2NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        if (this->param_propagate_down_[0]){

            const Dtype *normail_feature_data = Normalise_feature_.cpu_data();
            const Dtype *normail_weight_data = Normalise_Weight_.cpu_data();
            const Dtype* bottom_data = bottom[0]->cpu_data();
            const Dtype* top_diff = top[0]->cpu_diff();
            Dtype *normail_feature_diff = Normalise_feature_.mutable_cpu_diff();
            Dtype *normail_weight_diff = Normalise_Weight_.mutable_cpu_diff();
            Dtype * weight_diff = this->blobs_[0]->mutable_cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            
            /***********Gradient with respect to normalize weight******/
            caffe_cpu_gemm(CblasTrans, CblasNoTrans, Num_Class_, feature_Dim_, Num_BatchSize_,
                (Dtype)scaler_, top_diff, normail_feature_data, (Dtype)1., normail_weight_diff);
            /***********Gradient with respect to normalize feature******/
            caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, Num_BatchSize_, feature_Dim_, Num_Class_,
                (Dtype)scaler_, top_diff, normail_weight_data, (Dtype)0., normail_feature_diff);
            /**********background normalize bottom data feature*****************/
            for (int i=0; i<Num_BatchSize_; ++i) {
                Dtype a = caffe_cpu_dot(feature_Dim_, normail_feature_data+i*feature_Dim_, normail_feature_diff+i*feature_Dim_);
                caffe_cpu_scale(feature_Dim_, a, normail_feature_data+i*feature_Dim_, bottom_diff+i*feature_Dim_);
                caffe_sub(feature_Dim_, normail_feature_diff+i*feature_Dim_, bottom_diff+i*feature_Dim_, bottom_diff+i*feature_Dim_);
                a = caffe_cpu_dot(feature_Dim_, bottom_data+i*feature_Dim_, bottom_data+i*feature_Dim_);
                caffe_cpu_scale(feature_Dim_, Dtype(pow(a, -0.5)), bottom_diff+i*feature_Dim_, bottom_diff+i*feature_Dim_);
            }
            /**********background normalize weight*****************************/
            for (int i=0; i<Num_Class_; ++i) {
                Dtype a = caffe_cpu_dot(feature_Dim_, normail_weight_data+i*feature_Dim_, normail_weight_diff+i*feature_Dim_);
                caffe_cpu_scale(feature_Dim_, a, normail_weight_data+i*feature_Dim_, weight_diff+i*feature_Dim_);
                caffe_sub(feature_Dim_, normail_weight_diff+i*feature_Dim_, weight_diff+i*feature_Dim_, weight_diff+i*feature_Dim_);
                a = caffe_cpu_dot(feature_Dim_, this->blobs_[0]->cpu_data()+i*feature_Dim_, this->blobs_[0]->cpu_data()+i*feature_Dim_);
                caffe_cpu_scale(feature_Dim_, Dtype(pow(a, -0.5)), weight_diff+i*feature_Dim_, weight_diff+i*feature_Dim_);
            }
        }
        if (propagate_down[1]) {
            LOG(FATAL) << this->type()
                    << " Layer cannot backpropagate to label inputs.";
        }        
    }
    #ifdef CPU_ONLY
    STUB_GPU(CosinL2NormalizeLayer);
    #endif

    INSTANTIATE_CLASS(CosinL2NormalizeLayer);
    REGISTER_LAYER_CLASS(CosinL2Normalize);
}//name caffe