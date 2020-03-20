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
void ImageDataLayer<Dtype>::get_random_erasing_box(float sl, float sh, float min_rate, 
                                float max_rate, cv::Mat img, float *mean_value){
  CHECK_GE(sh, sl);
  CHECK_GT(sl, 0.);
  CHECK_LE(sh, 1.);

  CHECK_GE(max_rate, min_rate);
  CHECK_GT(min_rate, 0.);
  CHECK_LT(max_rate, 1);


  float scale;
  caffe_rng_uniform(1, sl, sh, &scale);

  // Get random aspect ratio.

  int height = img.rows;
  int width = img.cols;

  float aspect_ratio;
  caffe_rng_uniform(1, min_rate, max_rate, &aspect_ratio);

  aspect_ratio = std::max<float>(aspect_ratio, std::pow(scale, 2.));
  aspect_ratio = std::min<float>(aspect_ratio, 1 / std::pow(scale, 2.));

  float bbox_width = scale * sqrt(aspect_ratio);
  float bbox_height = scale / sqrt(aspect_ratio);

  // Figure out top left coordinates.
  float w_off, h_off;
  caffe_rng_uniform(1, 0.f, 1 - bbox_width, &w_off);
  caffe_rng_uniform(1, 0.f, 1 - bbox_height, &h_off);

  int xmin = int(w_off*width);
  int ymin = int(h_off*height);
  int xmax = int(bbox_width*width) + int(w_off*width);
  int ymax = int(bbox_height*height) + int(h_off*height);
  for(int row = ymin; row < ymax; row++){
    uchar* pdata= img.ptr<uchar>(row);
    for(int col = xmin; col < xmax; col++){
      pdata[col*3 + 0] = mean_value[0];
      pdata[col*3 + 1] = mean_value[1];
      pdata[col*3 + 2] = mean_value[2];
    }
  }
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
  CHECK_EQ(label_num_*sample_num_, this->layer_param_.image_data_param().batch_size());

  problity_ = this->layer_param_.image_data_param().probability();
  max_aspect_ratio_ = this->layer_param_.image_data_param().max_aspect_ratio();
  min_aspect_ratio_ = this->layer_param_.image_data_param().min_aspect_ratio();
  scale_lower_ = this->layer_param_.image_data_param().lower();
  scale_higher_ = this->layer_param_.image_data_param().higher();

  for(int ii = 0; ii < this->layer_param_.transform_param().mean_value().size(); ii++){
    mean_value[ii] = this->layer_param_.transform_param().mean_value(ii);
  }


  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  /**************遍历人脸数据集根目录文件夹**********/
  #if 0
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
  #endif

  const string &sourceMap = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening labelmap file: "<< sourceMap;
  std::ifstream infile(sourceMap.c_str());
  string line;
  size_t pos;
  int label;
  while(std::getline(infile, line)){
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos+1).c_str());
    std::string newDirectory = root_folder + string("/") + line.substr(0, pos);
    fullImageSetDir_.push_back(std::make_pair(newDirectory, label));
    //LOG(INFO)<< "directory: " << newDirectory << ", label: " << label;
  }
  infile.close();
  LOG(INFO)<<" fullImageSetDir_ size: " << fullImageSetDir_.size() << ", get file directory successfully";
  /**************遍历人脸数据集根目录文件夹完成*******/

  lines_id_ = 0;
    /**************获取第一个图像*************/
  std::string subDir = fullImageSetDir_[0].first;
  std::string imgfile;
  struct dirent *faceSetDir;
  DIR* dir = opendir(subDir.c_str());
  if( dir == NULL )
    LOG(FATAL) << subDir << " is not a directory or not exist!";
  while ((faceSetDir = readdir(dir)) != NULL) {
    if(strcmp(faceSetDir->d_name,".")==0 || strcmp(faceSetDir->d_name,"..")==0)
      continue;
    else if(faceSetDir->d_name[0] == '.')
      continue;
    else if (faceSetDir->d_type == DT_REG) {
      imgfile = subDir + string("/") + string(faceSetDir->d_name);
      break;
    }
  }
  closedir(dir);
  cv::Mat cv_img = ReadImageToCVMat(imgfile,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << imgfile;
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
  // sample_label_
  vector<int> label_shape_sample(1, batch_size);
  top[2]->Reshape(label_shape_sample);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].labelSample_.Reshape(label_shape_sample);
  }
}


// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(pairBatch<Dtype>* batch) {
  choosedImagefile_.clear();
  labelIdxSet_.clear();
  label.clear();
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

  /**************随机挑选符合要求的人脸图片*************/
  struct dirent *faceSetDir;
  std::vector<std::string> filelist;
  while (choosedImagefile_.size() < batch_size){
    int rand_class_idx = caffe_rng_rand() % fullImageSetDir_.size();
    while(std::count(labelIdxSet_.begin(), labelIdxSet_.end(), rand_class_idx)!=0){
      rand_class_idx = caffe_rng_rand() % fullImageSetDir_.size();
    }
    std::string subDir = fullImageSetDir_[rand_class_idx].first;
    filelist.clear();
    DIR* dir = opendir(subDir.c_str());
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
    unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count ();  
    std::shuffle(filelist.begin(), filelist.end(), std::default_random_engine (seed));
    for(int i = 0; i < nrof_image_from_class; i++){
      choosedImagefile_.push_back(std::make_pair(filelist[i], fullImageSetDir_[rand_class_idx].second));
    }
    labelIdxSet_.push_back(rand_class_idx);
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

  label_num_ = labelIdxSet_.size();
  vector<int> label_shape(1, label_num_);
  batch->label_.Reshape(label_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* labelSample = batch->labelSample_.mutable_cpu_data();
  

  // datum scales
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    float sampleProb = 0.0f;
    caffe_rng_uniform(1, 0.0f, 1.0f, &sampleProb);
    // get a blob
    timer.Start();
    cv::Mat cv_img = ReadImageToCVMat(choosedImagefile_[item_id].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << choosedImagefile_[item_id].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    if(sampleProb > problity_){ 
      // Apply transformations (mirror, crop...) to the image
      get_random_erasing_box(scale_lower_, scale_higher_, min_aspect_ratio_, 
                                max_aspect_ratio_, cv_img, mean_value);
    }
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();
    labelSample[item_id] = choosedImagefile_[item_id].second;
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
