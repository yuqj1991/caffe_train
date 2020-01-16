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
typedef struct _CenterNetInfo{
    int class_id;
    float score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
}CenterNetInfo;

template<typename Dtype>
Dtype gaussian_radius(const Dtype heatmap_width, const Dtype heatmap_height, const Dtype min_overlap);
template <typename Dtype>
void EncodeCenteGroundTruthAndPredictions(const Dtype* loc_data, const int output_width, 
                                const int output_height, 
                                bool share_location, Dtype* pred_data, const int num_channels,
                                Dtype* gt_data, std::map<int, vector<NormalizedBBox> > all_gt_bboxes);
template <typename Dtype>
void CopyDiffToBottom(const Dtype* pre_diff, const int output_width, 
                                const int output_height, 
                                bool share_location, Dtype* bottom_diff, const int num_channels,
                                std::map<int, vector<NormalizedBBox> > all_gt_bboxes);
template <typename Dtype>
void _nms_heatmap(const Dtype* conf_data, Dtype* keep_max_data, const int output_height
                  , const int output_width, const int channels, const int num_batch); 

template <typename Dtype>
void get_topK(const Dtype* keep_max_data, const Dtype* loc_data, const int output_height
                  , const int output_width, const int channels, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo> > results
                  , const int loc_channels);       

#ifdef USE_OPENCV
template <typename Dtype>
void transferCVMatToBlobData(cv::Mat heatmap, Dtype* buffer_heat);
cv::Mat gaussian2D(const int height, const int width, const float sigma);
void draw_umich_gaussian(cv::Mat heatmap, int center_x, int center_y, float radius, int k );

template <typename Dtype>
void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, Dtype* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);

#endif



}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
