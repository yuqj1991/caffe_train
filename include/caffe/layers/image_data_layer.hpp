#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDataLayer : public ImageDataPrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : ImageDataPrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void load_batch(pairBatch<Dtype>* batch);
  virtual void get_random_erasing_box(float sl, float sh, float min_rate, 
                                float max_rate, cv::Mat img, float *mean_value);

  int lines_id_;
  std::vector< std::pair<std::string, int> > fullImageSetDir_;
  std::vector< std::pair<std::string, int> > choosedImagefile_;
  int sample_num_;
  int label_num_;
  std::vector< int > labelIdxSet_;
  std::vector< int > label;
  float problity_;
  float max_aspect_ratio_;
  float min_aspect_ratio_;
  float scale_lower_;
  float scale_higher_;
  float mean_value[3];
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
