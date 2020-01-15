#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#ifndef CAFFE_UTIL_BBOX_UTIL_H_CENTER_HEATMAP_
#define CAFFE_UTIL_BBOX_UTIL_H_CENTER_HEATMAP_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

namespace caffe {
template<typename Dtype>
Dtype gaussian_radius(const int heatmap_width, const int heatmap_height, const Dtype min_overlap);

#ifdef USE_OPENCV

cv::Mat gaussian2D(const int height, const int width, const float sigma);
void draw_umich_gaussian(cv::Mat heatmap, float center_x, float center_y, float radius, int k ); 

#endif



}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
