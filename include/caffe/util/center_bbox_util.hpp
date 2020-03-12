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

#define NMS_UNION 1
#define NMS_MIN  2

template<typename Dtype>
Dtype gaussian_radius(const Dtype heatmap_height, const Dtype heatmap_width, const Dtype min_overlap);
template <typename Dtype>
void EncodeCenteGroundTruthAndPredictions(Dtype* gt_loc_data, Dtype* pred_loc_data,
                                const int output_width, const int output_height, 
                                bool share_location, const Dtype* channel_loc_data,
                                const int num_channels, std::map<int, vector<NormalizedBBox> > all_gt_bboxes);
template <typename Dtype>
void CopyDiffToBottom(const Dtype* pre_diff, const int output_width, 
                                const int output_height, 
                                bool share_location, Dtype* bottom_diff, const int num_channels,
                                std::map<int, vector<NormalizedBBox> > all_gt_bboxes);
template <typename Dtype>
void _nms_heatmap(const Dtype* conf_data, Dtype* keep_max_data, const int output_height
                  , const int output_width, const int channels, const int num_batch);

void nms(std::vector<CenterNetInfo>& input, std::vector<CenterNetInfo>* output, float nmsthreshold = 0.3,int type=NMS_MIN);

template <typename Dtype>
void get_topK(const Dtype* keep_max_data, const Dtype* loc_data, const int output_height
                  , const int output_width, const int channels, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > > * results
                  , const int loc_channels,  Dtype conf_thresh, Dtype nms_thresh);      

template <typename Dtype>
void transferCVMatToBlobData(std::vector<Dtype> heatmap, Dtype* buffer_heat);

template <typename Dtype>
std::vector<Dtype> gaussian2D(const int height, const int width, Dtype sigma);

template <typename Dtype>
void draw_umich_gaussian(std::vector<Dtype>& heatmap, int center_x, int center_y, float radius,const int height, const int width);

template <typename Dtype>
void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, Dtype* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);


template <typename Dtype>
void EncodeYoloGroundTruthAndPredictions(Dtype* gt_loc_data, Dtype* pred_loc_data,
                                const int output_width, const int output_height, 
                                bool share_location, const Dtype* channel_loc_data,
                                const int num_channels, std::map<int, vector<NormalizedBBox> > all_gt_bboxes);




template <typename Dtype>
void EncodeYoloObject(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int net_width, const int net_height,
                          const Dtype* channel_pred_data,
                          std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                          std::vector<int> mask_bias, std::vector<std::pair<int, int> >bias_scale, 
                          Dtype* bottom_diff, Dtype ignore_thresh);


}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
