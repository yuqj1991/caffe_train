#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void BatchNormScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    BatchNormParameter param = this->layer_param_.batch_norm_param();
    moving_average_fraction_ = param.moving_average_fraction();
    use_global_stats_ = this->phase_ == TEST;
    if (param.has_use_global_stats())
        use_global_stats_ = param.use_global_stats();
    if (bottom[0]->num_axes() == 1)
        channels_ = 1;
    else
        channels_ = bottom[0]->shape(1);
    eps_ = param.eps();
    
    /***************scale-parameter****************/
    // scale is a learned parameter; initialize it
    const ScaleParameter& scale_param = this->layer_param_.scale_param();
    axis_ = bottom[0]->CanonicalAxisIndex(scale_param.axis());
    const int num_axes = scale_param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                        << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
        CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
            << "scale blob's shape extends past bottom[0]'s shape when applied "
            << "starting with bottom[0] axis = " << axis_;
    }
    
    /***************scale-parameter****************/

    if (this->blobs_.size() > 0) {
        LOG(INFO) << "Skipping parameter initialization";
    } else {
        CHECK_EQ(scale_param.bias_term(), true);
        this->blobs_.resize(5);
        vector<int> sz;
        sz.push_back(channels_);
        this->blobs_[0].reset(new Blob<Dtype>(sz));
        this->blobs_[1].reset(new Blob<Dtype>(sz));
        sz[0] = 1;
        this->blobs_[2].reset(new Blob<Dtype>(sz));
        for (int i = 0; i < 3; ++i) {
        caffe_set(this->blobs_[i]->count(), Dtype(0),
                    this->blobs_[i]->mutable_cpu_data());
        }
        /***************scale-parameter****************/
        const vector<int>::const_iterator& shape_start =
            bottom[0]->shape().begin() + axis_;
        const vector<int>::const_iterator& shape_end =
            (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
        vector<int> scale_shape(shape_start, shape_end);
        this->blobs_[3].reset(new Blob<Dtype>(scale_shape));
        this->blobs_[4].reset(new Blob<Dtype>(scale_shape));
        FillerParameter filler_param(scale_param.filler());
        if (!scale_param.has_filler()) {
            // Default to unit (1) filler for identity operation.
            filler_param.set_type("constant");
            filler_param.set_value(1);
        }
        shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
        filler->Fill(this->blobs_[3].get());
        FillerParameter bias_filler_param(scale_param.bias_filler());
        if (!scale_param.has_bias_filler()) {
            // Default to unit (0) filler for identity operation.
            bias_filler_param.set_type("constant");
            bias_filler_param.set_value(0);
        }
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(bias_filler_param));
        bias_filler->Fill(this->blobs_[4].get());
        /***************scale-parameter****************/
    }
    // the mean, variance, bias_correction can not be learned,but the scale-parameter shoule be set true
    this->param_propagate_down_.resize(this->blobs_.size(), false);
    this->param_propagate_down_[3] = true;
    this->param_propagate_down_[4] = true;
    // Mask statistics from optimization by setting local learning rates
    // for mean, variance, and the bias correction(i.a.moving) to zero. 
    // and to set the scale-parameter the lr-policy(1., 2.)
    for (int i = 0; i < this->blobs_.size(); ++i) {
        if(i < 3){
            if (this->layer_param_.param_size() == i) {
                ParamSpec* fixed_param_spec = this->layer_param_.add_param();
                fixed_param_spec->set_lr_mult(0.f);
            } else {
                CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
                        << "Cannot configure batch normalization statistics as layer "
                        << "parameters.";
            }
        }
    }
}

template <typename Dtype>
void BatchNormScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    if (bottom[0]->num_axes() >= 1)
        CHECK_EQ(bottom[0]->shape(1), channels_);
    top[0]->ReshapeLike(*bottom[0]);

    vector<int> sz;
    sz.push_back(channels_);
    mean_.Reshape(sz);
    variance_.Reshape(sz);
    
    x_norm_.ReshapeLike(*bottom[0]);
    sz[0] = bottom[0]->shape(0);
    batch_sum_multiplier_.Reshape(sz);

    int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
    if (spatial_sum_multiplier_.num_axes() == 0 ||
        spatial_sum_multiplier_.shape(0) != spatial_dim) {
        sz[0] = spatial_dim;
        spatial_sum_multiplier_.Reshape(sz);
        Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
        caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
    }

    int numbychans = channels_*bottom[0]->shape(0);
    if (num_by_chans_.num_axes() == 0 ||
        num_by_chans_.shape(0) != numbychans) {
        sz[0] = numbychans;
        num_by_chans_.Reshape(sz);
        caffe_set(batch_sum_multiplier_.count(), Dtype(1),
            batch_sum_multiplier_.mutable_cpu_data());
    }

    /***************scale-Reshape****************/
    Blob<Dtype>* scale = this->blobs_[3].get();
    // Always set axis_ == 0 in special case where scale is a scalar
    // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
    // actual setting can be safely ignored; and computation is most efficient
    // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
    // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
    // performant.)
    axis_ = (scale->num_axes() == 0) ?
        0 : bottom[0]->CanonicalAxisIndex(this->layer_param_.scale_param().axis());
    CHECK_GE(bottom[0]->num_axes(), axis_ + scale->num_axes())
        << "scale blob's shape extends past bottom[0]'s shape when applied "
        << "starting with bottom[0] axis = " << axis_;
    for (int i = 0; i < scale->num_axes(); ++i) {
        CHECK_EQ(bottom[0]->shape(axis_ + i), scale->shape(i))
            << "dimension mismatch between bottom[0]->shape(" << axis_ + i
            << ") and scale->shape(" << i << ")";
    }
    outer_dim_ = bottom[0]->count(0, axis_);
    scale_dim_ = scale->count();
    inner_dim_ = bottom[0]->count(axis_ + scale->num_axes());
    bias_multiplier_.Reshape(vector<int>(1, inner_dim_));
    if (bias_multiplier_.cpu_data()[inner_dim_ - 1] != Dtype(1)) {
        caffe_set(inner_dim_, Dtype(1), bias_multiplier_.mutable_cpu_data());
    }
    /***************scale-Reshape****************/
}

template <typename Dtype>
void BatchNormScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int num = bottom[0]->shape(0);
    int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);

    if (bottom[0] != top[0]) {
        caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }

    if (use_global_stats_) {
        // use the stored mean/variance estimates.
        // 如果使用已经计算好的mean和variance
        // mean保存在blobs_[0]中，variance保存在blobs_[1]中
        // 滑动平均系数保存在blobs_[2]中
        const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
            0 : 1 / this->blobs_[2]->cpu_data()[0];
        caffe_cpu_scale(variance_.count(), scale_factor,
            this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
        caffe_cpu_scale(variance_.count(), scale_factor,
            this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
    } else {
        // 训练阶段  compute mean
        // 1.计算均值,先计算HW的，每个通道的均值，在包含N
        // caffe_cpu_gemv 实现 y =  alpha*A*x+beta*y;
        // 输出的是channels_*num,
        // 每次处理的列是spatial_dim，由于spatial_sum_multiplier_初始为1，即NCHW中的
        // H*W各自相加，得到N*C*average，此处多除以了num，下一步可以不除以。
        // compute mean
        //这个矩阵与向量相乘，目的是计算每个feature map的数值和，然后在除以1./(num*spatial_dim)
        //bottom_data: (channels_*num) x (spatial_dim)
        //spatial_sum_multiplier: spatial_dim x 1
        //alpha : 1./(num*spatial_dim); beta : 0
        //num_by_chans = alpha * (bottom_data x spatial_sum_multiplier) + beta * num_by_chans
        //其中spatial_sum_multiplier的值都为1
        //注意关键字是CblasTrans！！
        //num_by_chans_ : channels_ x num;
        //batch_sum_multiplier_ : num x 1;
        //mean_ = 1. x (num_by_chans_ x batch_sum_multiplier_)
        //mean_ : channels_ x 1
        //计算得到对应channels的平均值，这也解释了为什么之前要除以1./(num*spatial_dim)
        //而不是仅除以1./spatial_dim，这样减少了计算量
        caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
            1. / (num * spatial_dim), bottom_data,
            spatial_sum_multiplier_.cpu_data(), 0.,
            num_by_chans_.mutable_cpu_data());
        // 2.计算均值，计算N各的平均值.
        // 由于输出的是channels上的均值，因此需要转置
        // 上一步得到的N*C的均值，再按照num求均值，因为batch_sum全部为1,
        caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
            num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
            mean_.mutable_cpu_data());
    }

    // subtract mean
    // 进行 x - mean_x 操作，需要注意按照通道，即先确定x属于哪个通道.
    // caffe_cpu_gemm 实现alpha * A*B + beta* C
    // 输入是num*1 * 1* channels_,输出是num*channels_
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
        spatial_dim, 1, -1, num_by_chans_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), 1., top_data);

    if (!use_global_stats_) { 
        // compute variance using var(X) = E((X-EX)^2) 训练时，计算方差， 此处的top已经为x-mean_x了   
        Dtype* var_data = variance_.mutable_cpu_data();
        for(int c = 0; c < channels_; c++){
            Dtype sum_value = Dtype(0.);
            for(int b = 0; b < num; b++){
                for(int i = 0; i < spatial_dim; i++){
                    Dtype squre_value = Dtype(0.);
                    caffe_powx(1, top_data + b * channels_ * spatial_dim + c * spatial_dim + i, 
                                                            Dtype(2.0), &squre_value);
                    sum_value += squre_value;
                }
            }
            var_data[c] = sum_value / (num * spatial_dim);
        }
        // compute and save moving average
        // 均值和方差计算完成后，需要更新batch的滑动系数
        // y = alpha * x + beta * y
        // this->blobs_[2] 存放的是平均滑动系数
        this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
        this->blobs_[2]->mutable_cpu_data()[0] += 1;
        caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
            moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
        int m = bottom[0]->count()/channels_;
        Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
        caffe_cpu_axpby(variance_.count(), bias_correction_factor,
            variance_.cpu_data(), moving_average_fraction_,
            this->blobs_[1]->mutable_cpu_data());
    }

    // normalize variance, 方差求个根号,加上eps为防止分母为0
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
    caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
                variance_.mutable_cpu_data());

    // top_data = x-mean_x/sqrt(variance_),此处的top_data已经之前转化为x-mean_x了
    const Dtype* var_data = variance_.cpu_data();
    for(int b = 0; b < num; b ++){
        for(int c = 0; c < channels_; c++){
            caffe_cpu_scale(spatial_dim, Dtype(1 / var_data[c]), 
                                top_data + b * channels_ * spatial_dim + c * spatial_dim,
                                top_data + b * channels_ * spatial_dim + c * spatial_dim);
        }
    }
    caffe_copy(x_norm_.count(), top_data,
       x_norm_.mutable_cpu_data());// x_norm_.cpu_data stored normolized_top_data
    /********************scale-forward**************/
    const Dtype* scale_data = this->blobs_[3].get()->cpu_data();
    const Dtype* bias_data = this->blobs_[4].get()->cpu_data();
    for (int n = 0; n < outer_dim_; ++n) {
        for (int d = 0; d < scale_dim_; ++d) {
            const Dtype factor = scale_data[d];
            const Dtype bias = bias_data[d];
            caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
            caffe_add_scalar(inner_dim_, bias, top_data);
            bottom_data += inner_dim_;
            top_data += inner_dim_;
        }
    }
    /*
    caffe_copy(x_norm_.count(), top_data,
       x_norm_.mutable_cpu_diff()); // x_norm_.cpu_diff stored scaled_top_data
    */
    /********************scale-forward**************/
}

template <typename Dtype>
void BatchNormScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff;
    if (bottom[0] != top[0]) {
        top_diff = top[0]->cpu_diff();
    } else {
        top_diff = top[0]->cpu_diff();
    }  
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->shape()[0];
    int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
    const Dtype* norm_data = x_norm_.cpu_data();
    //const Dtype* top_data = x_norm_.cpu_diff();
    
    const Dtype* var_data = variance_.cpu_data();
    if (use_global_stats_) {
        for(int b = 0; b < num; b ++){
            for(int c = 0; c < channels_; c++){
                caffe_cpu_scale(spatial_dim, Dtype(1 / var_data[c]), 
                                    top_diff + b * channels_ * spatial_dim + c * spatial_dim,
                                    bottom_diff + b * channels_ * spatial_dim + c * spatial_dim);
            }
        }
        return;
    }
        
    // if S = alpha * (X-mean(X))/(sqrt(var(X)+eps)) + bias, then
    //
    // dE(S)/dX =
    //   alpha * (dE/dS - mean(dE/dS) - mean(dE/dS \cdot Y) \cdot Y)
    //     ./ sqrt(var(X) + eps)
    //
    // where \cdot and ./ are hadamard product and elementwise division,
    // respectively, dE/dS is the top diff, and mean/var/sum are all computed
    // along all dimensions except the channels dimension.  In the above
    // equation, the operations allow for expansion (i.e. broadcast) along all
    // dimensions except the channels dimension where required.

    // sum(dE/dS \cdot Y)
    caffe_mul(top[0]->count(), norm_data, top_diff, bottom_diff);
    //new_added
    /*****************scale-diff*************/
    if(this->param_propagate_down_[3]){
        Dtype* scale_diff = this->blobs_[3].get()->mutable_cpu_diff();
        for (int n = 0; n < outer_dim_; ++n) {
            caffe_cpu_gemv(CblasNoTrans, scale_dim_, inner_dim_, Dtype(1),
                bottom_diff, bias_multiplier_.cpu_data(), Dtype(1.), scale_diff);
            bottom_diff += scale_dim_ * inner_dim_;
        }
        
    }
    /*****************scale-diff*************/
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
        bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
        mean_.mutable_cpu_data());

    // reshape (broadcast) the above
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
        spatial_dim, 1, 1., num_by_chans_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

    // sum(dE/dS \cdot Y) \cdot Y
    caffe_mul(top[0]->count(), norm_data, bottom_diff, bottom_diff);

    // sum(dE/dS)-sum(dE/dS \cdot Y) \cdot Y
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
        top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
        mean_.mutable_cpu_data());
    // reshape (broadcast) the above to make
    // sum(dE/dS)-sum(dE/dS \cdot Y) \cdot Y
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
        spatial_dim, 1, 1., num_by_chans_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

    // dE/dS - mean(dE/dS)-mean(dE/dS \cdot Y) \cdot Y
    caffe_cpu_axpby(top[0]->count(), Dtype(1), top_diff,
        Dtype(-1. / (num * spatial_dim)), bottom_diff);

    // new added
    for(int b = 0; b < num; b ++){
        for(int c = 0; c < channels_; c++){
            caffe_cpu_scale(spatial_dim, Dtype(1 / var_data[c]), 
                                bottom_diff + b * channels_ * spatial_dim + c * spatial_dim,
                                bottom_diff + b * channels_ * spatial_dim + c * spatial_dim);
        }
    }
    const Dtype* scale_data = this->blobs_[3].get()->cpu_data();
    for (int n = 0; n < outer_dim_; ++n) {
        for (int d = 0; d < scale_dim_; ++d) {
            const Dtype factor = scale_data[d];
            caffe_cpu_scale(inner_dim_, factor, bottom_diff, bottom_diff);
            bottom_diff += inner_dim_;
        }
    }

    if(this->param_propagate_down_[4]){
        Dtype* bias_diff = this->blobs_[4].get()->mutable_cpu_diff();
        const bool bias_param = (bottom.size() == 1);
        bool accum = bias_param;
        for (int n = 0; n < outer_dim_; ++n) {
            caffe_cpu_gemv(CblasNoTrans, scale_dim_, inner_dim_, Dtype(1),
                top_diff, bias_multiplier_.cpu_data(), Dtype(accum), bias_diff);
            top_diff += scale_dim_ * inner_dim_;
            accum = true;
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(BatchNormScaleLayer);
#endif

INSTANTIATE_CLASS(BatchNormScaleLayer);
REGISTER_LAYER_CLASS(BatchNormScale);
}  // namespace caffe
