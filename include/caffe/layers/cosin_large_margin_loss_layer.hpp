#ifndef CAFFE_COSINE_LOSS_LAYER_HPP_
#define CAFFE_COSINE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
    template <typename Dtype>
    class CosinLossLayer : public LossLayer<Dtype>{
        public:
            explicit CosinLossLayer()(const LayerParameter& param)
                : LossLayer<Dtype>(param) {}
            virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
            virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
              virtual inline const char* type() const { return "CosinLoss"; }
            virtual inline int ExactNumBottomBlobs() const { return 2; }
            virtual inline int ExactNumTopBlobs() const { return -1; }
        protected:
            virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top); // 前向传播 CPU 实现
            virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top); // 前向传播 GPU 实现
            virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom); // 反向传播 CPU 实现
            virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);  //反向传播 GPU 实现
            inline Blob<Dtype> vector_L2_Normalise(Blob<Dtype>* srcData){
                const Dtype * data = srcData->cpu_data();
                Blob<Dtype> Distdata;
                Distdata.ReshapeLike(&srcData);
                Dtype * distData = Distdata.mutable_cpu_data();
                int num = srcData->num();
                int dim = srcData->count(1);
                Dtype sum_squre;
                for (size_t i = 0; i < num; i++)
                {
                    caffe_powx(dim, data + i * dim, Dtype(2.0), distData + i*dim);
                    sum_squre = caffe_cpu_asum(dim, distData + i*dim);
                    caffe_cpu_axpby(dim, Dtype(1.0/caffe_sqrt(sum_squre)), data + i * dim, Dtype(0.0),distData + i * dim);
                }
                return Distdata;
            }

        Blob<Dtype> Normalise_Weight_;
        Blob<Dtype> Normalise_feature_;
        
        int Num_Class_;
        int Num_BatchSize_;
        int feature_Dim_;
        float margin_;
        float scaler_;
        shared_ptr<Layer<Dtype> > conf_loss_layer_;
        // bottom vector holder used in Forward function.
        vector<Blob<Dtype>*> conf_bottom_vec_;
        // top vector holder used in Forward function.
        vector<Blob<Dtype>*> conf_top_vec_;
        // blob which stores the confidence prediction.
        Blob<Dtype> Normalizer_cosValue_;
        // blob which stores the corresponding ground truth label.
        Blob<Dtype> conf_gt_;
        // confidence loss.
        Blob<Dtype> conf_loss_;
    }
} //namespace cafe

#endif // CAFFE_COSINE_LOSS_LAYER_HPP_
