#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BatchNormScaleForward(const int n, const Dtype* in,
    const Dtype* scale, const int scale_dim, const int inner_dim,
    Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
        const int scale_index = (index / inner_dim) % scale_dim;
        out[index] = in[index] * scale[scale_index];
    }
}

template <typename Dtype>
__global__ void BatchNormScaleBiasForward(const int n, const Dtype* in,
    const Dtype* scale, const Dtype* bias,
    const int scale_dim, const int inner_dim, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index] + bias[scale_index];
    }
}


template <typename Dtype>
__global__ void batchNorm_forward(int nthreads, int width, int height, int channels, 
                                  Dtype* top_data, const Dtype* bottom_data, const Dtype* var_data){
    CUDA_KERNEL_LOOP(index, nthreads){
        const int fc = (index / width / height) % channels;
        top_data[index] = bottom_data[index] / var_data[fc];
    }
}

template <typename Dtype>
void BatchNormScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int num = bottom[0]->shape(0);
    int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));

    if (bottom[0] != top[0]) {
        caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }
    if (use_global_stats_) {
        // use the stored mean/variance estimates.
        const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
            0 : 1 / this->blobs_[2]->cpu_data()[0];
        caffe_gpu_scale(variance_.count(), scale_factor,
            this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
        caffe_gpu_scale(variance_.count(), scale_factor,
            this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());
    } else {
        // compute mean
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
            1. / (num * spatial_dim), bottom_data,
            spatial_sum_multiplier_.gpu_data(), 0.,
            num_by_chans_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
            num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
            mean_.mutable_gpu_data());
    }
    // subtract mean, top_data = x - mean(x)
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
        spatial_dim, 1, -1, num_by_chans_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 1., top_data);
    
    int nthreads = bottom[0]->count();
    int width = bottom[0]->width();
    int height = bottom[0]->height();
    if (!use_global_stats_) {
        // compute variance using var(X) = E((X-EX)^2)
        
        caffe_gpu_powx(top[0]->count(), top_data, Dtype(2),
            x_norm_.mutable_gpu_data());  // (X-EX)^2
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
            1. / (num * spatial_dim), x_norm_.gpu_data(),
            spatial_sum_multiplier_.gpu_data(), 0.,
            num_by_chans_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
            num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
            variance_.mutable_gpu_data());  // E((X_EX)^2)
                
        // compute and save moving average
        this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
        this->blobs_[2]->mutable_cpu_data()[0] += 1;
        caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
            moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
        int m = bottom[0]->count()/channels_;
        Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
        caffe_gpu_axpby(variance_.count(), bias_correction_factor,
            variance_.gpu_data(), moving_average_fraction_,
            this->blobs_[1]->mutable_gpu_data());
    }
    // normalize variance
    caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
    caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
        variance_.mutable_gpu_data());

    batchNorm_forward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
        width, height, channels_, top_data, 
        top[0]->gpu_data(), variance_.gpu_data());

    caffe_copy(x_norm_.count(), top_data,
        x_norm_.mutable_gpu_data());// x_norm_.gpu_data stored normolized_top_data
    /********************scale-forward**************/
    const Dtype* scale_data = this->blobs_[3]->gpu_data();
    const Dtype* bias_data = this->blobs_[4]->gpu_data();
    const int count = top[0]->count();
    BatchNormScaleBiasForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_data, bias_data, channels_, inner_dim_,
        top_data);
    /*
    caffe_copy(x_norm_.count(), top_data,
        x_norm_.mutable_gpu_diff()); // x_norm_.gpu_diff stored scaled_top_data
    */
    /********************scale-forward**************/
}

template <typename Dtype>
__global__ void batchNorm_backward(int nthreads, int width, int height, int channels, 
                                  const Dtype* x, const Dtype* var_data, Dtype *y, const Dtype* scale_data){
    CUDA_KERNEL_LOOP(index, nthreads){
        const int fc = (index / width / height) % channels;
        y[index] = x[index] * scale_data[fc] / var_data[fc];
    }
}

template <typename Dtype>
__global__ void batchNorm_backward_param(int nthreads, int width, int height, int channels, 
    const Dtype* x, Dtype *y){
        CUDA_KERNEL_LOOP(index, nthreads){
            const int fc = (index / width / height) % channels;
            y[fc] += x[index];
        }
}

template <typename Dtype>
void BatchNormScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* norm_data = x_norm_.gpu_data();
    int num = bottom[0]->shape()[0];
    int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
    int nthreads = bottom[0]->count();
    int width = bottom[0]->width();
    int height = bottom[0]->height();
    const Dtype* top_diff;
    if (bottom[0] != top[0]) {
        top_diff = top[0]->gpu_diff();
    } else {
        caffe_copy(x_norm_.count(), top[0]->gpu_diff(), x_norm_.mutable_gpu_diff());
        top_diff = top[0]->gpu_diff();
    }
    #if 1
    if(this->param_propagate_down_[4]){
        Dtype* bias_diff = this->blobs_[4]->mutable_gpu_diff();
        #if 0
        bool accum = true;
        for (int n = 0; n < outer_dim_; ++n) {
            caffe_gpu_gemv(CblasNoTrans, channels_, spatial_dim, Dtype(1),
                top_diff, spatial_sum_multiplier_.gpu_data(), Dtype(accum), bias_diff);
            top_diff += channels_ * spatial_dim;
            accum = true;
        }
        #else
        batchNorm_backward_param<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
            width, height, channels_, top_diff, bias_diff);
        #endif
    }
    #endif
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    //const Dtype* scale_data = this->blobs_[3]->gpu_data();
    if (use_global_stats_) {
        batchNorm_backward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
            width, height, channels_, bottom_diff, variance_.gpu_data(), bottom_diff
            , this->blobs_[3]->gpu_data());
        return;
    }
    
      
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
    caffe_gpu_mul(top[0]->count(), norm_data, top_diff, bottom_diff);
    //new_added
    #if 1
    /*****************scale-diff*************/
    if(this->param_propagate_down_[3]){
        Dtype* scale_diff = this->blobs_[3]->mutable_gpu_diff();
        batchNorm_backward_param<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
            width, height, channels_, bottom_diff, scale_diff);
    }
    #endif
    /*****************scale-diff*************/
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
        bottom_diff, spatial_sum_multiplier_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
        mean_.mutable_gpu_data());

    // reshape (broadcast) the above
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
        spatial_dim, 1, 1., num_by_chans_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 0., bottom_diff);

    // sum(dE/dY \cdot Y) \cdot Y
    caffe_gpu_mul(top[0]->count(), norm_data, bottom_diff, bottom_diff);

    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
        top_diff, spatial_sum_multiplier_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
        mean_.mutable_gpu_data());
    // reshape (broadcast) the above to make
    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
        spatial_dim, 1, 1., num_by_chans_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 1., bottom_diff);

    // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
    caffe_gpu_axpby(top[0]->count(), Dtype(1), top_diff, Dtype(-1. / (num * spatial_dim)), bottom_diff);
    // new added
    batchNorm_backward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
                    width, height, channels_, bottom_diff, variance_.gpu_data(), bottom_diff, 
                    this->blobs_[3]->gpu_data());
    
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormScaleLayer);

}  // namespace caffe