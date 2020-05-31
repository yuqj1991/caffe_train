#ifndef CAFFE_UTIL_BBOX_UTIL_H_CENTER_HEATMAP_SINGLE_
#define CAFFE_UTIL_BBOX_UTIL_H_CENTER_HEATMAP_SINGLE_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "caffe/util/center_util.hpp"
#include "caffe/caffe.hpp"

namespace caffe {

template <typename Dtype>
void GetLocTruthAndPrediction(Dtype* gt_loc_offest_data, Dtype* pred_loc_offest_data,
                                Dtype* gt_loc_wh_data, Dtype* pred_loc_wh_data,
                                Dtype* gt_lm_data, Dtype* pred_lm_data,
                                const int output_width, const int output_height, 
                                bool share_location, const Dtype* loc_data,
                                const Dtype* wh_data, const Dtype* lm_data,
                                const int loc_channels, const int lm_channels, 
                                std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                                bool has_lm);
template <typename Dtype>
void CopySingleDiffToBottom(const Dtype* pre_offset_diff, const Dtype* pre_wh_diff, const int output_width, 
                                const int output_height, bool has_lm, const Dtype* lm_pre_diff,
                                bool share_location, Dtype* loc_diff,
                                Dtype* wh_diff, Dtype* lm_diff,
                                const int loc_channels, const int lm_channels,
                                std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes);

template <typename Dtype>
void get_single_topK(const Dtype* keep_max_data, const Dtype* loc_data
                  , const Dtype* wh_data, const Dtype* lm_data,
                  const int loc_channels, const int lm_channels, 
                  , const int output_height
                  , const int output_width, const int channels, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > > * results
                  , const int loc_channels, bool has_lm,  Dtype conf_thresh, Dtype nms_thresh);

}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
