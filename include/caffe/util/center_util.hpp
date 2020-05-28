#ifndef CAFFE_UTIL_H_CENTER_
#define CAFFE_UTIL_H_CENTER_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

namespace caffe {

float YoloBBoxIou(NormalizedBBox a, NormalizedBBox b);

int int_index(std::vector<int>a, int val, int n);

template <typename Dtype>
Dtype CenterSigmoid(Dtype x);

template <typename Dtype>
Dtype SingleSoftmaxLoss(Dtype bg_score, Dtype face_score, Dtype lable_value);

template <typename T>
bool SortScorePairDescendCenter(const pair<T, float>& pair1,
                          const pair<T, float>& pair2);


template <typename Dtype>
Dtype smoothL1_Loss(Dtype x, Dtype* x_diff);

template <typename Dtype>
Dtype L2_Loss(Dtype x, Dtype* x_diff);

template <typename Dtype>
Dtype Object_L2_Loss(Dtype x, Dtype* x_diff);

template<typename Dtype>
Dtype gaussian_radius(const Dtype heatmap_height, const Dtype heatmap_width, const Dtype min_overlap);


template <typename Dtype>
void transferCVMatToBlobData(std::vector<Dtype> heatmap, Dtype* buffer_heat);

template <typename Dtype>
std::vector<Dtype> gaussian2D(const int height, const int width, Dtype sigma);

template <typename Dtype>
void draw_umich_gaussian(std::vector<Dtype>& heatmap, int center_x, int center_y, float radius,const int height, const int width);


template <typename Dtype>
void SelectHardSampleSoftMax(Dtype *label_data, std::vector<Dtype> batch_sample_loss,
                          const int negative_ratio, std::vector<int> postive, 
                          const int output_height, const int output_width,
                          const int num_channels, const int batch_size);

template <typename Dtype>
void SoftmaxCenterGrid(Dtype * pred_data, const int batch_size,
            const int label_channel, const int num_channels,
            const int outheight, const int outwidth);

template <typename Dtype>
Dtype SoftmaxLossEntropy(Dtype* label_data, Dtype* pred_data, 
                            const int batch_size, const int output_height, 
                            const int output_width, Dtype *bottom_diff, 
                            const int num_channels);
template <typename Dtype>
void SelectHardSampleSigmoid(Dtype *label_data, Dtype *pred_data, const int negative_ratio, const int num_postive, 
                          const int output_height, const int output_width, const int num_channels);

template <typename Dtype> 
Dtype FocalLossSigmoid(Dtype* label_data, Dtype * pred_data, int dimScale, Dtype *bottom_diff);

template <typename Dtype>
Dtype GIoULoss(NormalizedBBox predict_box, NormalizedBBox gt_bbox, Dtype* diff_x1, 
                Dtype* diff_x2, Dtype* diff_y1, Dtype* diff_y2, const int anchor_scale,
                const int downRatio, const int layer_scale);

template <typename Dtype>
Dtype DIoULoss(NormalizedBBox predict_box, NormalizedBBox gt_bbox, Dtype* diff_x1, 
                Dtype* diff_x2, Dtype* diff_y1, Dtype* diff_y2);

}  // namespace caffe

#endif  