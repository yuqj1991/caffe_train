#include "caffe/layers/sample_triplet_layer.hpp"
#include "math.h"

namespace caffe {

template <typename Dtype>
void SampleTripletLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  label_num_ = bottom[1]->count();
  batch_size_ = bottom[0]->num();
  triplet_num_ = label_num_ ;
  top[0]->Reshape(label_num_, 3, 1, 1);
  feature_dim_ = bottom[0]->count(1);
  inner_matrix_.Reshape(batch_size_, batch_size_, 1, 1);
}

template <typename Dtype>
void SampleTripletLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  alpha_ = this->layer_param_.sample_triplet_param().alpha();
  Dtype *top_data = top[0]->mutable_cpu_data();

  /*************************我自己添加的***********************/
  const Dtype *feature_data = bottom[0]->cpu_data();  //feature data
  const Dtype *label_data = bottom[1]->cpu_data(); //label data
  int emb_start_idx = 0;
  int neg_images = 0;
  an_set.clear();
  positive_set.clear();
  neg_set.clear();
  int j = 0;
  for(int i = 0; i < label_num_; i++){
    int nrof_images = (int)label_data[i];
    neg_images = batch_size_ - nrof_images;
    for(j = 1; j < nrof_images; j++){
      neg_dist_sqr.clear();
      int a_idx = emb_start_idx + j - 1;
      /**********计算neg距离****************/
      for(int n = 0; n < batch_size_; n++){
        int min_anchor_idx = emb_start_idx;
        int max_anchor_idx = emb_start_idx + nrof_images;
        if(n >= min_anchor_idx && n < max_anchor_idx){
          continue;
        }else{
          float neg_sum = 0.f;
          for(int ii = 0; ii < feature_dim_; ii++){
            float dim = std::pow(feature_data[a_idx*feature_dim_ +ii] - feature_data[n*feature_dim_ +ii], 2);
            neg_sum += dim;
          }
          neg_dist_sqr.push_back(std::make_pair(n, neg_sum));
        }
      }
      /***********计算postive距离**********/
      for(int pair = j; pair < nrof_images; pair++ ){
          int p_idx = emb_start_idx + pair;
          float pos_dist_sqr =0.f;
          for(int ii = 0; ii < feature_dim_; ii++){
            float dim = std::pow(feature_data[a_idx*feature_dim_ +ii] - feature_data[p_idx*feature_dim_ +ii], 2);
            pos_dist_sqr += dim;
          }
          vector<int> all_neg;
          for(int nn = 0; nn < neg_images; nn++){
            float diff = neg_dist_sqr[nn].second - pos_dist_sqr;
            if(diff < alpha_){
              all_neg.push_back(neg_dist_sqr[nn].first);
            }
          }
          if(all_neg.size() > 0){
            int rand_idx = caffe_rng_rand() % all_neg.size();
            int n_idx = all_neg[rand_idx];
            an_set.push_back(a_idx);
            positive_set.push_back(p_idx);
            neg_set.push_back(n_idx);
          }
      }
    }
    emb_start_idx += nrof_images;
  }
  triplet_num_ = neg_set.size();
  top[0]->Reshape(triplet_num_, 3, 1, 1);
  for(int idx = 0; idx<triplet_num_; idx++){
    top_data[idx * 3] = an_set[idx];
    top_data[idx * 3 + 1] = positive_set[idx];
    top_data[idx * 3 + 2] = neg_set[idx];
  }

}

INSTANTIATE_CLASS(SampleTripletLayer);
REGISTER_LAYER_CLASS(SampleTriplet);


}