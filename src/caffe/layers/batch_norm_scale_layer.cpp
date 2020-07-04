#include <algorithm>
#include <vector>

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
    has_scale_param_ = this->layer_param_.has_scale_param();
    if(has_scale_param_){
        const ScaleParameter& scale_param = this->layer_param_.scale_param();
        if (bottom.size() == 1) {
            // scale is a learned parameter; initialize it
            axis_ = bottom[0]->CanonicalAxisIndex(scale_param.axis());
            const int num_axes = scale_param.num_axes();
            CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                                << "or -1 to extend to the end of bottom[0]";
            if (num_axes >= 0) {
                CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
                    << "scale blob's shape extends past bottom[0]'s shape when applied "
                    << "starting with bottom[0] axis = " << axis_;
            }
        }
        bias_term_ = scale_param.bias_term();
    }

    if (this->blobs_.size() > 0) {
        LOG(INFO) << "Skipping parameter initialization";
    } else {
        this->blobs_.resize(3 + (has_scale_param_ ? ( 1 + bias_term_ ? 1: 0): 0));
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
        if(has_scale_param_){
            const ScaleParameter& scale_param = this->layer_param_.scale_param();
            const vector<int>::const_iterator& shape_start =
                bottom[0]->shape().begin() + axis_;
            const vector<int>::const_iterator& shape_end =
                (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
            vector<int> scale_shape(shape_start, shape_end);
            this->blobs_[3].reset(new Blob<Dtype>(scale_shape));
            FillerParameter filler_param(scale_param.filler());
            if (!scale_param.has_filler()) {
                // Default to unit (1) filler for identity operation.
                filler_param.set_type("constant");
                filler_param.set_value(1);
            }
            shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
            filler->Fill(this->blobs_[3].get());
            if(bias_term_){
                LayerParameter layer_param(this->layer_param_);
                layer_param.set_type("Bias");
                BiasParameter* bias_param = layer_param.mutable_bias_param();
                bias_param->set_axis(scale_param.axis());
                if (bottom.size() > 1) {
                    bias_param->set_num_axes(bottom[1]->num_axes());
                } else {
                    bias_param->set_num_axes(scale_param.num_axes());
                }
                bias_param->mutable_filler()->CopyFrom(scale_param.bias_filler());
                bias_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
                bias_bottom_vec_.resize(1);
                bias_bottom_vec_[0] = bottom[0];
                bias_layer_->SetUp(bias_bottom_vec_, top);
                if (this->blobs_.size() + bottom.size() < 3) {
                    // case: blobs.size == 1 && bottom.size == 1
                    // or blobs.size == 0 && bottom.size == 2
                    bias_param_id_ = this->blobs_.size();
                    this->blobs_.resize(bias_param_id_ + 1);
                    this->blobs_[bias_param_id_] = bias_layer_->blobs()[0];
                } else {
                    // bias param already initialized
                    bias_param_id_ = this->blobs_.size() - 1;
                    bias_layer_->blobs()[0] = this->blobs_[bias_param_id_];
                }
                bias_propagate_down_.resize(1, false);
            }
        }
    }
    // Mask statistics from optimization by setting local learning rates
    // for mean, variance, and the bias correction to zero.
    for (int i = 0; i < this->blobs_.size(); ++i) {
        if(i < 3){
            if (this->layer_param_.param_size() == i) {
                ParamSpec* fixed_param_spec = this->layer_param_.add_param();
                fixed_param_spec->set_lr_mult(0.f);
            } else {
                CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
                    << "Cannot configure batch normalization & scale statistics as layer "
                    << "parameters.";
            }
        }
    }
    this->param_propagate_down_.resize(this->blobs_.size(), false);
    if(has_scale_param_){
        this->param_propagate_down_[this->blobs_.size() - 1 - (bias_term_ ? 1 : 0)] = true;
        if(bias_term_)
            this->param_propagate_down_[this->blobs_.size() - 1] = true;
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
    temp_.ReshapeLike(*bottom[0]);
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
    if(has_scale_param_){
        const ScaleParameter& scale_param = this->layer_param_.scale_param();
        Blob<Dtype>* scale = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
        // Always set axis_ == 0 in special case where scale is a scalar
        // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
        // actual setting can be safely ignored; and computation is most efficient
        // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
        // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
        // performant.)
        axis_ = (scale->num_axes() == 0) ?
            0 : bottom[0]->CanonicalAxisIndex(scale_param.axis());
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
        if (bottom[0] == top[0]) {  // in-place computation
            temp_.ReshapeLike(*bottom[0]);
        } else {
            top[0]->ReshapeLike(*bottom[0]);
        }
        sum_result_.Reshape(vector<int>(1, outer_dim_ * scale_dim_));
        const int sum_mult_size = std::max(outer_dim_, inner_dim_);
        sum_multiplier_.Reshape(vector<int>(1, sum_mult_size));
        if (sum_multiplier_.cpu_data()[sum_mult_size - 1] != Dtype(1)) {
            caffe_set(sum_mult_size, Dtype(1), sum_multiplier_.mutable_cpu_data());
        }
        if (bias_layer_) {
            bias_bottom_vec_[0] = top[0];
            bias_layer_->Reshape(bias_bottom_vec_, top);
        }
    }
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
    // 此处的均值已经保存在mean_中了
    // 进行 x - mean_x 操作，需要注意按照通道，即先确定x属于哪个通道.
    // 因此也是进行两种，先进行H*W的减少均值
    // caffe_cpu_gemm 实现alpha * A*B + beta* C
    // 输入是num*1 * 1* channels_,输出是num*channels_
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
        spatial_dim, 1, -1, num_by_chans_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), 1., top_data);

    if (!use_global_stats_) {
        // compute variance using var(X) = E((X-EX)^2)
        // 测试的方差上述已经读取了
        // compute variance using var(X) = E((X-EX)^2)
        // 此处的top已经为x-mean_x了
        caffe_powx(top[0]->count(), top_data, Dtype(2),
            temp_.mutable_cpu_data());  // (X-EX)^2
        // 同均值一样，此处先计算spatial_dim的值
        caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
            1. / (num * spatial_dim), temp_.cpu_data(),
            spatial_sum_multiplier_.cpu_data(), 0.,
            num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
            num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
            variance_.mutable_cpu_data());  // E((X_EX)^2)

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

    // normalize variance 
    // 方差求个根号,加上eps为防止分母为0
    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
    caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
                variance_.mutable_cpu_data());

    // replicate variance to input size
    // top_data = x-mean_x/sqrt(variance_),此处的top_data已经转化为x-mean_x了
    // 同减均值，也要分C--N*C和  N*C --- N*C*H*W
    // N*1 *  1*C == N*C
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
        spatial_dim, 1, 1., num_by_chans_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
    // temp最终保存的是sqrt（方差+eps)
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
    // TODO(cdoersch): The caching is only needed because later in-place layers
    //                 might clobber the data.  Can we skip this if they won't?
    caffe_copy(x_norm_.count(), top_data,
        x_norm_.mutable_cpu_data());
}

template <typename Dtype>
void BatchNormScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff;
    if (bottom[0] != top[0]) {
        top_diff = top[0]->cpu_diff();
    } else {
        caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
        top_diff = x_norm_.cpu_diff();
    }
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (use_global_stats_) {
        caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
        return;
    }
    const Dtype* top_data = x_norm_.cpu_data();
    int num = bottom[0]->shape()[0];
    int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
    // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
    //
    // dE(Y)/dX =
    //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
    //     ./ sqrt(var(X) + eps)
    //
    // where \cdot and ./ are hadamard product and elementwise division,
    // respectively, dE/dY is the top diff, and mean/var/sum are all computed
    // along all dimensions except the channels dimension.  In the above
    // equation, the operations allow for expansion (i.e. broadcast) along all
    // dimensions except the channels dimension where required.

    // sum(dE/dY \cdot Y)
    caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
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

    // sum(dE/dY \cdot Y) \cdot Y
    caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
        top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
        mean_.mutable_cpu_data());
    // reshape (broadcast) the above to make
    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
        spatial_dim, 1, 1., num_by_chans_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

    // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
    caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
        Dtype(-1. / (num * spatial_dim)), bottom_diff);

    // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
    // pass.
    caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(BatchNormScaleLayer);
#endif

INSTANTIATE_CLASS(BatchNormScaleLayer);
REGISTER_LAYER_CLASS(BatchNormScale);
}  // namespace caffe
