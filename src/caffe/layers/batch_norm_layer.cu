#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void batchNorm_variance(int nthreads, int width, int height, int channels, const Dtype* bottom_data, Dtype* top_data,Dtype* var_data){
    CUDA_KERNEL_LOOP(index, nthreads){
        const int fc = (index / width / height) % channels;
        printf("bottom_data: %lf, top_data: %lf", bottom_data[index], top_data[index]);
        var_data[fc] += pow(top_data[index], 2.);
    }
}

template <typename Dtype>
__global__ void batchNorm_forward(int nthreads, int width, int height, int channels, 
                                  Dtype* top_data, const Dtype* var_data){
    CUDA_KERNEL_LOOP(index, nthreads){
        const int fc = (index / width / height) % channels;
        top_data[index] = top_data[index] / var_data[fc];
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
        batchNorm_variance<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
            width, height, channels_, bottom_data,
            top_data, variance_.mutable_gpu_data());

        caffe_gpu_scale(variance_.count(), Dtype(1. / (num * spatial_dim)),
            variance_.gpu_data(), variance_.mutable_gpu_data());
        
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
    // new added
    
    batchNorm_forward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
                                width, height, channels_, 
                                top_data, variance_.gpu_data());  
}

template <typename Dtype>
__global__ void batchNorm_backward(int nthreads, int width, int height, int channels, 
                                  const Dtype* top_diff, Dtype* bottom_diff, const Dtype* var_data){
    CUDA_KERNEL_LOOP(index, nthreads){
        const int fc = (index / width / height) % channels;
        bottom_diff[index] = top_diff[index] / var_data[fc];
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff;
    if (bottom[0] != top[0]) {
        top_diff = top[0]->gpu_diff();
    } else {
        top_diff = top[0]->gpu_diff();
    }
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    const Dtype* top_data = top[0]->gpu_data();
    int num = bottom[0]->shape()[0];
    int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
    const Dtype* var_data = variance_.gpu_data();

    //New added
    int nthreads = bottom[0]->count();
    int width = bottom[0]->width();
    int height = bottom[0]->height();
    if (use_global_stats_) {
        //new added
        batchNorm_backward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
            width, height, channels_, top_diff, bottom_diff, var_data);
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
    caffe_gpu_mul(top[0]->count(), top_data, top_diff, bottom_diff); // top[0]-> temp_
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
    caffe_gpu_mul(top[0]->count(), top_data, bottom_diff, bottom_diff); // top[0]-> temp_

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
    caffe_gpu_axpby(top[0]->count(), Dtype(1), top_diff, Dtype(-1. / (num * spatial_dim)), bottom_diff); // top[0]-> temp_
    // new add
    batchNorm_backward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
                    width, height, channels_, bottom_diff, bottom_diff, var_data);
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormLayer);

}  // namespace caffe
