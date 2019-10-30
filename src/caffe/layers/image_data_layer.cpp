#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <random> // std::default_random_engine
#include <chrono> // std::chrono::system_clock


#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


#include <dirent.h>
#include <stdio.h>


namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  sample_num_ =  this->layer_param_.image_data_param().sample_num();
  label_num_= this->layer_param_.image_data_param().label_num();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  CHECK_EQ(label_num_*sample_num_, this->layer_param_.data_param().batch_size());

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines_.push_back(std::make_pair(line.substr(0, pos), label));
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, label_num_);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  /**************遍历人脸数据集根目录文件夹**********/
  struct dirent *faceSetDir;
  DIR* dir = opendir(root_folder.c_str());
  if( dir == NULL )
    LOG(FATAL)<<" is not a directory or not exist!";
    
  while ((faceSetDir = readdir(dir)) != NULL) {
      if(strcmp(faceSetDir->d_name,".")==0 || strcmp(faceSetDir->d_name,"..")==0)    ///current dir OR parrent dir
        continue;
      else if(faceSetDir->d_name[0] == '.')
        continue;
      else if (faceSetDir->d_type == DT_DIR) {
        std::string newDirectory = root_folder + string("/") + string(faceSetDir->d_name);
        fullImageSetDir_.push_back(newDirectory);
      }
  }
  closedir(dir);
  /**************随机挑选符合要求的人脸图片*************/
  while (choosedImagefile_.size()<batch_size){
    int rand_class_idx = caffe_rng_rand() % fullImageSetDir_.size();
    while(std::count(labelSet_.begin(), labelSet_.end(), rand_class_idx)!=0){
      rand_class_idx = caffe_rng_rand() % fullImageSetDir_.size();
    }
    std::string subDir = fullImageSetDir_[rand_class_idx];
    std::vector<std::string> filelist;
    dir = opendir(subDir.c_str());
    if( dir == NULL )
      LOG(FATAL)<<" is not a directory or not exist!";
    while ((faceSetDir = readdir(dir)) != NULL) {
      if(strcmp(faceSetDir->d_name,".")==0 || strcmp(faceSetDir->d_name,"..")==0)
        continue;
      else if(faceSetDir->d_name[0] == '.')
        continue;
      else if (faceSetDir->d_type == DT_REG) {
        std::string imgfile = subDir + string("/") + string(faceSetDir->d_name);
        filelist.push_back(imgfile);
      }
    }
    closedir(dir);
    int nrof_image_in_class = filelist.size();
    int length = choosedImagefile_.size();
    int temp = std::min(nrof_image_in_class, batch_size - length );
    int nrof_image_from_class = std::min(sample_num_, temp);
    caffe::rng_t* prefetch_rng =
                            static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(filelist.begin(), filelist.end(), prefetch_rng);
    for(int i = 0; i < nrof_image_from_class; i++){
      choosedImagefile_.push_back(std::make_pair(filelist[i], rand_class_idx));
    }
    labelSet_.push_back(rand_class_idx);
    label.push_back(nrof_image_from_class);
  }

  /**************遍历人脸数据集根目录遍历文件夹**********/

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(choosedImagefile_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << choosedImagefile_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  label_num_ = labelSet_.size();
  vector<int> label_shape(1, label_num_);
  batch->label_.Reshape(label_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  

  // datum scales
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    //CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(choosedImagefile_[item_id].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << choosedImagefile_[item_id].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();
    #if 0 // 前人做的
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
    #endif
  }
  for(int i = 0; i < label_num_; i++){
    prefetch_label[i] = label[i];
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
#endif  // USE_OPENCV
