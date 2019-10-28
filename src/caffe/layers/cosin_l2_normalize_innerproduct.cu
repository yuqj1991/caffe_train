#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/cosin_l2_normalize_innerproduct.hpp"

namespace caffe {
    template <typename Dtype>
    void CosinL2NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        this->Forward_cpu(bottom, top);
    }

    template <typename Dtype>
    void CosinL2NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        this->Backward_cpu(top, propagate_down, bottom);
    }
    INSTANTIATE_LAYER_GPU_FUNCS(CosinL2NormalizeLayer);
}
