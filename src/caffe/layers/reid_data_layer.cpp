#include <stdint.h>
#include <cfloat>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/reid_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include <boost/thread.hpp>

namespace caffe {

template <typename Dtype>
ReidDataLayer<Dtype>::~ReidDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
unsigned int ReidDataLayer<Dtype>::RandRng() {
  CHECK(prefetch_rng_);
  caffe::rng_t *prefetch_rng =
      static_cast<caffe::rng_t *>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void ReidDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  DLOG(INFO) << "ReidDataLayer : DataLayerSetUp";

  // Main Data Layer Set up
  const int new_height = this->layer_param_.reid_data_param().new_height();
  const int new_width  = this->layer_param_.reid_data_param().new_width();
  const bool is_color  = this->layer_param_.reid_data_param().is_color();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.reid_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  int mx_label = -1;
  int mi_label = INT_MAX;
  while (std::getline(infile, line)) {
    size_t pos = line.find_last_of(' ');
    int label = atoi(line.substr(pos + 1).c_str());
    this->lines_.push_back(std::make_pair(line.substr(0, pos), label));
    mx_label = std::max(mx_label, label);
    mi_label = std::min(mi_label, label);
  }
  CHECK_EQ(mi_label, 0);
  this->label_set.clear();
  this->label_set.resize(mx_label+1);
  for (size_t index = 0; index < this->lines_.size(); index++) {
    int label = this->lines_[index].second;
    this->label_set[label].push_back(index);
  }
  for (size_t index = 0; index < this->label_set.size(); index++) {
    CHECK_GT(this->label_set[index].size(), 0) << "label : " << index << " has no images";
  }

  CHECK(!lines_.empty()) << "File is empty";
  infile.close();

  LOG(INFO) << "A total of " << lines_.size() << " images. Label : [" << mi_label << ", " << mx_label << "]";
  LOG(INFO) << "A total of " << label_set.size() << " persons";
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  this->left_images = this->lines_.size();
  this->pos_fraction = this->layer_param_.reid_data_param().pos_fraction();
  this->neg_fraction = this->layer_param_.reid_data_param().pos_fraction();

  CHECK_GT(lines_.size(), 0);

  this->cv_imgs_.clear();
  for (size_t lines_id_ = 0; lines_id_ < this->lines_.size(); lines_id_++) {
    cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_].first, new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    this->cv_imgs_.push_back(cv_img);
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(lines_[0].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[0].first;

  const int batch_size = this->layer_param_.reid_data_param().batch_size();

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  vector<int> prefetch_top_shape = top_shape;
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size * 2;
  prefetch_top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  //top[1]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(prefetch_top_shape);
    this->prefetch_[i]->datap_.Reshape(prefetch_top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  //LOG(INFO) << "output data pair size: " << top[1]->num() << ","
  //    << top[1]->channels() << "," << top[1]->height() << ","
  //    << top[1]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size*2);
    top[1]->Reshape(label_shape);
    vector<int> prefetch_label_shape(1, batch_size);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(prefetch_label_shape);
      this->prefetch_[i]->labelp_.Reshape(prefetch_label_shape);
    }
    LOG(INFO) << "output label size : " << top[1]->shape_string();
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void ReidDataLayer<Dtype>::load_batch(ReidBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.reid_data_param().batch_size();
  const vector<size_t> batches = this->batch_ids();
  const vector<size_t> batches_pair = this->batch_pairs(batches);

  CHECK_EQ(batches.size(), batch_size);
  CHECK_EQ(batches_pair.size(), batch_size);
  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = this->cv_imgs_[batches[0]];
  CHECK(cv_img.data) << "Could not load " << this->lines_[batches[0]].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  batch->datap_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_datap = batch->datap_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_labelp = batch->labelp_.mutable_cpu_data();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    const size_t true_idx = batches[item_id];
    const size_t pair_idx = batches_pair[item_id];
    cv::Mat cv_img_true = this->cv_imgs_[ true_idx ];
    cv::Mat cv_img_pair = this->cv_imgs_[ pair_idx ];
    CHECK(cv_img_true.data) << "Could not load " << this->lines_[true_idx].first;
    CHECK(cv_img_pair.data) << "Could not load " << this->lines_[pair_idx].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    const int t_offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + t_offset);
    this->data_transformer_->Transform(cv_img_true, &(this->transformed_data_));

    // Pair Data
    const int p_offset = batch->datap_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_datap + p_offset);
    this->data_transformer_->Transform(cv_img_pair, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    CHECK_GE(lines_[true_idx].second, 0);
    CHECK_GE(lines_[pair_idx].second, 0);
    CHECK_LT(lines_[true_idx].second, this->label_set.size());
    CHECK_LT(lines_[pair_idx].second, this->label_set.size());
    prefetch_label[item_id]    = lines_[true_idx].second;
    prefetch_labelp[item_id]   = lines_[pair_idx].second;

    DLOG(INFO) << "Idx : " << item_id << " : " << lines_[true_idx].second << " vs " << lines_[pair_idx].second;
  }
  batch_timer.Stop();
  DLOG(INFO) << "Pair Idx : (" << batches[0] << "," << batches_pair[0] << ")";
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ReidDataLayer);
REGISTER_LAYER_CLASS(ReidData);

}  // namespace caffe
