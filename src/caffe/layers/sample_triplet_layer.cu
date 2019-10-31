#include "caffe/layers/sample_triplet_layer.hpp"

namespace caffe {

template <typename Dtype>
void SampleTripletLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  alpha_ = this->layer_param_.sample_triplet_param().alpha();
  //Dtype eps = this->layer_param_.sample_triplet_param().eps();
  Dtype *top_data = top[0]->mutable_cpu_data();

  /*************************我自己添加的***********************/
  const Dtype *feature_data = bottom[0]->cpu_data();  //feature data
  const Dtype *label_data = bottom[1]->cpu_data(); //label data
  int nrof_images = 0;
  int emb_start_idx = 0;
  int neg_images = 0;
  an_set.clear();
  positive_set.clear();
  neg_set.clear();
  int j = 0;
  for(int i = 0; i < label_num_; i++){
    nrof_images = label_data[i];
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
  /*********************以上是我自己添加的****************************/
  /*
  int n_num = batch_size_ - sample_num_;
  int triplet_idx = 0;
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, batch_size_, batch_size_,
      feature_dim_, Dtype(1), bottom[0]->gpu_data(),
      bottom[0]->gpu_data(), Dtype(0), inner_matrix_.mutable_gpu_data());
  const Dtype *inner_data = inner_matrix_.cpu_data();
  for (int i = 0; i < label_num_; ++i) {
    int n_f = (i + 1) * sample_num_ % batch_size_;
    int n_r = (n_f + n_num) % batch_size_;
    for(int j = 0; j < sample_num_; ++j) {
      int a_idx = i * sample_num_ + j;
      int a_m_idx = a_idx * batch_size_ + a_idx;
      Dtype norm_a = sqrt(inner_data[a_m_idx] + eps);
      for (int k = 0; k < sample_num_; ++k) {
        if (k != j) {
          int p_idx = i * sample_num_ + k;
          int tmp_n_idx = n_f;
          int n_idx = -1;
          Dtype max_an = -1;
          while (tmp_n_idx != n_r) {
            int n_m_idx = tmp_n_idx * batch_size_ + tmp_n_idx;
            int an_m_idx = a_idx * batch_size_ + tmp_n_idx;
            Dtype norm_n = sqrt(inner_data[n_m_idx] + eps);
            Dtype tmp_an = inner_data[an_m_idx];
            tmp_an /= (norm_a * norm_n);
            if (tmp_an >= max_an) {
              max_an = tmp_an;
              n_idx = tmp_n_idx;
            }
            tmp_n_idx = (tmp_n_idx + 1) % batch_size_;
          }
          top_data[triplet_idx * 3] = a_idx;
          top_data[triplet_idx * 3 + 1] = p_idx;
          top_data[triplet_idx * 3 + 2] = n_idx;
          triplet_idx++;
        }
      }
    }
  }
  */
}

template <typename Dtype>
void SampleTripletLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}

INSTANTIATE_LAYER_GPU_FUNCS(SampleTripletLayer);

}
