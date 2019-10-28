#ifndef CAFFE_COSINE_NORMALIZEAL_LAYER_HPP_
#define CAFFE_COSINE_NORMALIZEAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    template <typename Dtype>
    class CosinL2NormalizeLayer : public Layer<Dtype>{
        public:
            explicit CosinL2NormalizeLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
            virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
            virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
              virtual inline const char* type() const { return "CosinL2Normalize"; }
            virtual inline int ExactNumBottomBlobs() const { return 1; }
            virtual inline int ExactNumTopBlobs() const { return 1; }
        protected:
            virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top); // 前向传播 CPU 实现
            virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top); // 前向传播 GPU 实现
            virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom); // 反向传播 CPU 实现
            virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);  //反向传播 GPU 实现
            inline void vector_L2_Normalise(const Dtype * data, int NumBatch, int featureDim, Dtype * distData){
                Dtype sum_squre;
                for (size_t i = 0; i < NumBatch; i++)
                {
                    caffe_powx(featureDim, data + i * featureDim, Dtype(2.0), distData + i*featureDim);
                    sum_squre = caffe_cpu_asum(featureDim, distData + i*featureDim) + 0.00000000001;
                    caffe_cpu_axpby(featureDim, Dtype(1.0/std::sqrt((double)sum_squre)), data + i * featureDim, Dtype(0.0),distData + i * featureDim);
                }
            }

        Blob<Dtype> Normalise_Weight_;
        Blob<Dtype> Normalise_feature_;
        
        int Num_Class_;
        int Num_BatchSize_;
        int feature_Dim_;
        float margin_;
        float scaler_;
        #if 0
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
        #endif
    };
} //namespace cafe

#endif // CAFFE_COSINE_LOSS_LAYER_HPP_
