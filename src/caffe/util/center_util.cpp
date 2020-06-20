#include <algorithm>
#include <csignal>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "boost/iterator/counting_iterator.hpp"

#include "caffe/util/bbox_util.hpp"
#include "caffe/util/center_util.hpp"


namespace caffe {
float Yoloverlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float boxIntersection(NormalizedBBox a, NormalizedBBox b)
{
    float a_center_x = (float)(a.xmin() + a.xmax()) / 2;
    float a_center_y = (float)(a.ymin() + a.ymax()) / 2;
    float a_w = (float)(a.xmax() - a.xmin());
    float a_h = (float)(a.ymax() - a.ymin());
    float b_center_x = (float)(b.xmin() + b.xmax()) / 2;
    float b_center_y = (float)(b.ymin() + b.ymax()) / 2;
    float b_w = (float)(b.xmax() - b.xmin());
    float b_h = (float)(b.ymax() - b.ymin());
    float w = Yoloverlap(a_center_x, a_w, b_center_x, b_w);
    float h = Yoloverlap(a_center_y, a_h, b_center_y, b_h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float boxUnion(NormalizedBBox a, NormalizedBBox b)
{
    float i = boxIntersection(a, b);
    float a_w = (float)(a.xmax() - a.xmin());
    float a_h = (float)(a.ymax() - a.ymin());
    float b_w = (float)(b.xmax() - b.xmin());
    float b_h = (float)(b.ymax() - b.ymin());
    float u = a_h*a_w + b_w*b_h - i;
    return u;
}

float YoloBBoxIou(NormalizedBBox a, NormalizedBBox b){
    return (float)boxIntersection(a, b)/boxUnion(a, b);
}

int int_index(std::vector<int>a, int val, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        if(a[i] == val) return i;
    }
    return -1;
}

template <typename Dtype>
Dtype CenterSigmoid(Dtype x){
	return 1. / (1. + exp(-x));
}

template double CenterSigmoid(double x);
template float CenterSigmoid(float x);

template <typename Dtype>
Dtype SingleSoftmaxLoss(Dtype bg_score, Dtype face_score, Dtype lable_value){
    Dtype Probability_value = Dtype(0.f);
    if(lable_value == 1.){
        Probability_value = face_score;
    }else{
        Probability_value = bg_score;
    }
    Dtype loss = (-1) * log(std::max(Probability_value,  Dtype(FLT_MIN)));
    return loss;
}

template float SingleSoftmaxLoss(float bg_score, float face_score, float lable_value);
template double SingleSoftmaxLoss(double bg_score, double face_score, double lable_value);

template <typename T>
bool SortScorePairDescendCenter(const pair<T, float>& pair1,
                          const pair<T, float>& pair2) {
    return pair1.second > pair2.second;
}

template <typename Dtype>
Dtype smoothL1_Loss(Dtype x, Dtype* x_diff){
    Dtype loss = Dtype(0.);
    Dtype fabs_x_value = std::fabs(x);
    if(fabs_x_value < 1){
        loss = 0.5 * x * x;
        *x_diff = x;
    }else{
        loss = fabs_x_value - 0.5;
        *x_diff = (Dtype(0) < x) - (x < Dtype(0));
    }
    return loss;
}
template float smoothL1_Loss(float x, float* x_diff);
template double smoothL1_Loss(double x, double* x_diff);

template <typename Dtype>
Dtype L2_Loss(Dtype x, Dtype* x_diff){
    Dtype loss = Dtype(0.);
    loss = x * x;
    *x_diff =2 * x;
    return loss;
}

template float L2_Loss(float x, float* x_diff);
template double L2_Loss(double x, double* x_diff);

template <typename Dtype>
Dtype Object_L2_Loss(Dtype x, Dtype* x_diff){
    Dtype loss = Dtype(0.);
    loss =0.5 * x * x;
    *x_diff =x;
    return loss;
}

template float Object_L2_Loss(float x, float* x_diff);
template double Object_L2_Loss(double x, double* x_diff);


template<typename Dtype>
Dtype gaussian_radius(const Dtype heatmap_height, const Dtype heatmap_width, const Dtype min_overlap){

    Dtype a1  = Dtype(1.0);
    Dtype b1  = (heatmap_width + heatmap_height);
    Dtype c1  = Dtype( heatmap_width * heatmap_height * (1 - min_overlap) / (1 + min_overlap));
    Dtype sq1 = std::sqrt(b1 * b1 - 4 * a1 * c1);
    Dtype r1  = Dtype((b1 + sq1) / 2);

    Dtype a2  = Dtype(4.0);
    Dtype b2  = 2 * (heatmap_height + heatmap_width);
    Dtype c2  = (1 - min_overlap) * heatmap_width * heatmap_height;
    Dtype sq2 = std::sqrt(b2 * b2 - 4 * a2 * c2);
    Dtype r2  = Dtype((b2 + sq2) / 2);

    Dtype a3  = Dtype(4 * min_overlap);
    Dtype b3  = -2 * min_overlap * (heatmap_height + heatmap_width);
    Dtype c3  = (min_overlap - 1) * heatmap_width * heatmap_height;
    Dtype sq3 = std::sqrt(b3 * b3 - 4 * a3 * c3);
    Dtype r3  = Dtype((b3 + sq3) / 2);
    return std::min(std::min(r1, r2), r3);
}

template float gaussian_radius(const float heatmap_width, const float heatmap_height, const float min_overlap);
template double gaussian_radius(const double heatmap_width, const double heatmap_height, const double min_overlap);


template<typename Dtype>
std::vector<Dtype> gaussian2D(const int height, const int width, Dtype sigma){
    int half_width = (width - 1) / 2;
    int half_height = (height - 1) / 2;
    std::vector<Dtype> heatmap((width *height), Dtype(0.));
    for(int i = 0; i < height; i++){
        int x = i - half_height;
        for(int j = 0; j < width; j++){
            int y = j - half_width;
            heatmap[i * width + j] = std::exp(float(-(x*x + y*y) / (2* sigma * sigma)));
            if(heatmap[i * width + j] < 0.00000000005)
                heatmap[i * width + j] = Dtype(0.);
        }
    }
    return heatmap;
}

template std::vector<float> gaussian2D(const int height, const int width, float sigma);
template std::vector<double> gaussian2D(const int height, const int width, double sigma);

template<typename Dtype>
void draw_umich_gaussian(std::vector<Dtype>& heatmap, int center_x, int center_y, float radius
                              , const int height, const int width){
    float diameter = 2 * radius + 1;
    std::vector<Dtype> gaussian = gaussian2D(int(diameter), int(diameter), Dtype(diameter / 6));
    int left = std::min(int(center_x), int(radius)), right = std::min(int(width - center_x), int(radius) + 1);
    int top = std::min(int(center_y), int(radius)), bottom = std::min(int(height - center_y), int(radius) + 1);
    if((left + right) > 0 && (top + bottom) > 0){
        for(int row = 0; row < (top + bottom); row++){
            for(int col = 0; col < (right + left); col++){
                int heatmap_index = (int(center_y) -top + row) * width + int(center_x) -left + col;
                int gaussian_index = (int(radius) - top + row) * int(diameter) + int(radius) - left + col;
                heatmap[heatmap_index] = heatmap[heatmap_index] >= gaussian[gaussian_index]  ? heatmap[heatmap_index]:
                                            gaussian[gaussian_index];
            }
        }
    }
}

template void draw_umich_gaussian(std::vector<float>& heatmap, int center_x, int center_y, float radius, const int height, const int width);
template void draw_umich_gaussian(std::vector<double>& heatmap, int center_x, int center_y, float radius, const int height, const int width);

template <typename Dtype>
void transferCVMatToBlobData(std::vector<Dtype> heatmap, Dtype* buffer_heat){
  for(unsigned ii = 0; ii < heatmap.size(); ii++){
        buffer_heat[ii] = buffer_heat[ii] > heatmap[ii] ? 
                                              buffer_heat[ii] : heatmap[ii];
  }
}
template void transferCVMatToBlobData(std::vector<float> heatmap, float* buffer_heat);
template void transferCVMatToBlobData(std::vector<double> heatmap, double* buffer_heat);



template <typename Dtype> 
Dtype FocalLossSigmoid(Dtype* label_data, Dtype * pred_data, int dimScale, Dtype *bottom_diff){
    Dtype alpha_ = 2.0f;
    Dtype gamma_ = 4.0f;
    Dtype loss = Dtype(0.);
    for(int i = 0; i < dimScale; i++){
        if(label_data[i] == 0.5){ // gt_boxes之外的负样本
            loss -= alpha_ * std::pow(pred_data[i], gamma_) * std::log(std::max(1 - pred_data[i], Dtype(FLT_MIN)));
            Dtype diff_elem_ = alpha_ * std::pow(pred_data[i], gamma_);
            Dtype diff_next_ = pred_data[i] - gamma_ * (1 - pred_data[i]) * std::log(std::max(1 - pred_data[i], Dtype(FLT_MIN)));
            bottom_diff[i] = diff_elem_ * diff_next_;
        }else if(label_data[i] == 1){ //gt_boxes包围的都认为是正样本
            loss -= alpha_ * std::pow(1 - pred_data[i], gamma_) * std::log(std::max(pred_data[i], Dtype(FLT_MIN)));
            Dtype diff_elem_ = alpha_ * std::pow(1 - pred_data[i], gamma_);
            Dtype diff_next_ = gamma_ * pred_data[i] * std::log(std::max(pred_data[i], Dtype(FLT_MIN))) + pred_data[i] - 1;
            bottom_diff[i] = diff_elem_ * diff_next_;
        }
    }
    return loss;
}

template float FocalLossSigmoid(float* label_data, float *pred_data, int dimScale,  float *bottom_diff);
template double FocalLossSigmoid(double* label_data, double *pred_data, int dimScale,  double *bottom_diff);


// hard sampling mine postive : negative 1: 5 sigmoid
// 按理来说是需要重新统计负样本的编号，以及获取到他的数值
// label_data : K x H x W
// pred_data : K x H x W x N
template <typename Dtype>
void SelectHardSampleSigmoid(Dtype *label_data, Dtype *pred_data, const int negative_ratio, const int num_postive, 
                          const int output_height, const int output_width, const int num_channels){
    CHECK_EQ(num_channels, 4 + 2) << "x, y, width, height + objectness + label class containing face";
    std::vector<std::pair<int, float> > loss_value_indices;
    loss_value_indices.clear();
    Dtype alpha_ = 0.25;
    Dtype gamma_ = 2.f;
    for(int h = 0; h < output_height; h ++){
        for(int w = 0; w < output_width; w ++){
            if(label_data[h * output_width +w] == 0.){
                int bg_index = h * output_width + w;
                // Focal loss when sample belong to background
                Dtype loss = (-1) * alpha_ * std::pow(pred_data[bg_index], gamma_) * 
                                            std::log(std::max(1 - pred_data[bg_index],  Dtype(FLT_MIN)));
                loss_value_indices.push_back(std::make_pair(bg_index, loss));
            }
        }
    }
    std::sort(loss_value_indices.begin(), loss_value_indices.end(), SortScorePairDescendCenter<int>);
    int num_negative = std::min(int(loss_value_indices.size()), num_postive * negative_ratio);
    for(int ii = 0; ii < num_negative; ii++){
        int h = loss_value_indices[ii].first / output_width;
        int w = loss_value_indices[ii].first % output_width;
        label_data[h * output_width + w] = 0.5;
    }
}
template void SelectHardSampleSigmoid(float *label_data, float *pred_data, const int negative_ratio, const int num_postive, 
                                     const int output_height, const int output_width, const int num_channels);
template void SelectHardSampleSigmoid(double *label_data, double *pred_data, const int negative_ratio, const int num_postive, 
                                     const int output_height, const int output_width, const int num_channels);



template<typename Dtype>
Dtype FocalLossSoftmax(Dtype* label_data, Dtype* pred_data, 
                            const int batch_size, const int output_height, 
                            const int output_width, Dtype *bottom_diff, 
                            const int num_channels, bool has_lm){
    Dtype loss = Dtype(0.f);
    float alpha = 0.25f;
    float gamma = 2.f;
    //float alpha = 2.f;
    //float gamma = 4.f;
    int dimScale = output_height * output_width;
    for(int b = 0; b < batch_size; b++){
        for(int h = 0; h < output_height; h++){
            for(int w = 0; w < output_width; w++){
                Dtype label_value = Dtype(label_data[b * dimScale + h * output_width + w]);
                if(label_value < 0.f){
                    continue;
                }else{
                    int label_idx = 0;
                    if(label_value == 0.5)
                        label_idx = 0;
                    else if(label_value == 1.)
                        label_idx = 1;
                    else{
                        LOG(FATAL)<<"no valid label value";
                    }
                    int bg_index = b * num_channels * dimScale + 4 * dimScale + h * output_width + w;
                    if(has_lm){
                        bg_index = b * num_channels * dimScale + 14 * dimScale + h * output_width + w;
                    }
                    Dtype p1 = pred_data[bg_index + label_idx * dimScale];
                    Dtype p0 = 1 - p1;

                    #if 0
                    loss -= alpha * std::pow(p0, gamma) * std::log(std::max(p1,  Dtype(FLT_MIN)));
                    bottom_diff[bg_index + label_idx * dimScale] = (alpha) * std::pow(p0, gamma) * 
                                                                (gamma * std::log(std::max(p1,  Dtype(FLT_MIN))) * p1 - p0);
                    bottom_diff[bg_index + (1 - label_idx) * dimScale] = (alpha) * std::pow(p0, gamma) * 
                                                                (p0 - gamma * std::log(std::max(p1,  Dtype(FLT_MIN))) * p1 );
                    #else
                    Dtype assist_value = Dtype((p0 + 1.) / (p1 + 1.));
                    Dtype p0_temp = Dtype(1. / (p0 + 1.));
                    Dtype p1_temp = Dtype(1. / (p1 + 1.));
                    Dtype p1_inverse = Dtype(1. / (p1 + 0.00000000000001));
                    loss -= alpha * std::pow(assist_value, gamma) * std::log(std::max(p1,  Dtype(FLT_MIN)));
                    bottom_diff[bg_index + label_idx * dimScale] = (alpha) * std::pow(assist_value, gamma) * p0 * p1 *
                                (p1_inverse - gamma * std::log(std::max(p1,  Dtype(FLT_MIN)) * (p0_temp + p1_temp)));
                    bottom_diff[bg_index + (1 - label_idx) * dimScale] = (alpha) * std::pow(assist_value, gamma) * p0 * p1 *
                                (gamma * std::log(std::max(p1,  Dtype(FLT_MIN)) * (p0_temp + p1_temp) - p1_inverse));
                    #endif
                }
            }
        }
    }
    return loss;
}
template float FocalLossSoftmax(float* label_data, float* pred_data, 
                            const int batch_size, const int output_height, 
                            const int output_width, float *bottom_diff, 
                            const int num_channels, bool has_lm);
template double FocalLossSoftmax(double* label_data, double* pred_data, 
                            const int batch_size, const int output_height, 
                            const int output_width, double *bottom_diff, 
                            const int num_channels, bool has_lm);

// label_data shape N : 1
// pred_data shape N : k (object classes)
// dimScale is the number of what ?? N * K ??
template <typename Dtype>
Dtype SoftmaxWithLoss(Dtype* label_data, Dtype* pred_data, 
                            const int batch_size, const int output_height, 
                            const int output_width, Dtype *bottom_diff, 
                            const int num_channels, bool has_lm){
    Dtype loss = Dtype(0.f);
    int dimScale = output_height * output_width;
    for(int b = 0; b < batch_size; b++){
        for(int h = 0; h < output_height; h++){
            for(int w = 0; w < output_width; w++){
                Dtype label_value = Dtype(label_data[b * dimScale + h * output_width + w]);
                if(label_value < 0.f){
                    continue;
                }else{
                    int label_idx = 0;
                    if(label_value == 0.5)
                        label_idx = 0;
                    else if(label_value == 1.)
                        label_idx = 1;
                    else{
                        LOG(FATAL)<<"no valid label value";
                    }
                    int bg_index = b * num_channels * dimScale + 4 * dimScale + h * output_width + w;
                    if(has_lm){
                        bg_index = b * num_channels * dimScale + 14 * dimScale + h * output_width + w;
                    }

                    Dtype Probability_value = pred_data[bg_index + label_idx * dimScale];
                    Dtype pred_another_data_value = pred_data[bg_index + (1 - label_idx) * dimScale];
                    loss -= log(std::max(Probability_value,  Dtype(FLT_MIN)));
                    bottom_diff[bg_index + label_idx * dimScale] = Probability_value - 1;
                    bottom_diff[bg_index + (1 - label_idx) * dimScale] = pred_another_data_value;
                }
            }
        }
    }
    return loss;
}

template float SoftmaxWithLoss(float* label_data, float* pred_data, 
                            const int batch_size, const int output_height, 
                            const int output_width, float *bottom_diff, 
                            const int num_channels, bool has_lm);
template double SoftmaxWithLoss(double* label_data, double* pred_data, 
                            const int batch_size, const int output_height, 
                            const int output_width, double *bottom_diff, 
                            const int num_channels, bool has_lm);

template <typename Dtype>
void SoftmaxCenterGrid(Dtype * pred_data, const int batch_size,
            const int label_channel, const int num_channels,
            const int outheight, const int outwidth, bool has_lm){
    int dimScale = outheight * outwidth;
    for(int b = 0; b < batch_size; b ++){
        for(int h = 0; h < outheight; h++){
            for(int w = 0; w < outwidth; w++){
                int bg_index = b * num_channels * dimScale + 4 * dimScale +  h * outwidth + w;
                if(has_lm)
                {
                    bg_index = b * num_channels * dimScale + 14 * dimScale +  h * outwidth + w;
                }
                Dtype MaxVaule = pred_data[bg_index + 0 * dimScale];
                Dtype sumValue = Dtype(0.f);
                // 求出每组的最大值
                for(int c = 0; c< label_channel; c++){
                    MaxVaule = std::max(MaxVaule, pred_data[bg_index + c * dimScale]);
                }
                // 每个样本组减去最大值， 计算exp，求和
                for(int c = 0; c< label_channel; c++){
                    pred_data[bg_index + c * dimScale] = std::exp(pred_data[bg_index + c * dimScale] - MaxVaule);
                    sumValue += pred_data[bg_index + c * dimScale];
                }
                // 计算softMax
                for(int c = 0; c< label_channel; c++){
                    pred_data[bg_index + c * dimScale] = Dtype(pred_data[bg_index + c * dimScale] / sumValue);
                }
            }
        }
    }
}

template void SoftmaxCenterGrid(float * pred_data, const int batch_size,
            const int label_channel, const int num_channels,
            const int outheight, const int outwidth, bool has_lm);
template void SoftmaxCenterGrid(double * pred_data, const int batch_size,
            const int label_channel, const int num_channels,
            const int outheight, const int outwidth, bool has_lm);
// hard sampling mine postive : negative 1: 5 softmax
// 按理来说是需要重新统计负样本的编号，以及获取到他的数值
// label_data : K x H x W
template <typename Dtype>
void SelectHardSampleSoftMax(Dtype *label_data, std::vector<Dtype> batch_sample_loss,
                          const int negative_ratio, std::vector<int> postive, 
                          const int output_height, const int output_width,
                          const int num_channels, const int batch_size, bool has_lm){
    if(has_lm){
        CHECK_EQ(num_channels, 14 + 2) << "x, y, width, height, landmarks + label classes containing background + face";
    }else{
        CHECK_EQ(num_channels, 4 + 2) << "x, y, width, height + label classes containing background + face";
    }
    CHECK_EQ(postive.size(), batch_size);
    int num_postive = 0;
    int dimScale = output_height * output_width;
    std::vector<std::pair<int, float> > loss_value_indices;
    #if 0
    loss_value_indices.clear();
    for(int b = 0; b < batch_size; b ++){
        num_postive += postive[b];
        for(int h = 0; h < output_height; h ++){
            for(int w = 0; w < output_width; w ++){
                int select_index = b * dimScale + h * output_width + w;
                if(label_data[select_index] == -1.){
                    loss_value_indices.push_back(std::make_pair(select_index, batch_sample_loss[select_index]));
                }
            }
        }
    }
    std::sort(loss_value_indices.begin(), loss_value_indices.end(), SortScorePairDescendCenter<int>);
    int num_negative = std::min(int(loss_value_indices.size()), num_postive * negative_ratio);
    for(int ii = 0; ii < num_negative; ii++){
        int select_index = loss_value_indices[ii].first;
        label_data[select_index] = 0.5;
    }
    #else    
    for(int b = 0; b < batch_size; b ++){
        num_postive = postive[b];
        loss_value_indices.clear();
        for(int h = 0; h < output_height; h ++){
            for(int w = 0; w < output_width; w ++){
                int select_index = b * dimScale + h * output_width + w;
                if(label_data[select_index] == -1.){
                    loss_value_indices.push_back(std::make_pair(select_index, batch_sample_loss[select_index]));
                }
            }
        }
        std::sort(loss_value_indices.begin(), loss_value_indices.end(), SortScorePairDescendCenter<int>);
        int num_negative = std::min(int(loss_value_indices.size()), num_postive * negative_ratio);
        for(int ii = 0; ii < num_negative; ii++){
            int select_index = loss_value_indices[ii].first;
            label_data[select_index] = 0.5;
            //LOG(INFO)<<"bg loss: "<<loss_value_indices[ii].second<<", label: "<<0;
        }
    }
    //LOG(INFO)<<"%%%%%%%%%%%%%%%%";
    #endif
}

template void SelectHardSampleSoftMax(float *label_data, std::vector<float> batch_sample_loss, 
                          const int negative_ratio, std::vector<int> postive, 
                          const int output_height, const int output_width,
                          const int num_channels, const int batch_size, bool has_lm);
template void SelectHardSampleSoftMax(double *label_data, std::vector<double> batch_sample_loss, 
                          const int negative_ratio, std::vector<int> postive, 
                          const int output_height, const int output_width,
                          const int num_channels, const int batch_size, bool has_lm);


template <typename Dtype>
Dtype GIoULoss(NormalizedBBox predict_box, NormalizedBBox gt_bbox, Dtype* diff_x1, 
                Dtype* diff_x2, Dtype* diff_y1, Dtype* diff_y2, const int anchor_scale,
                const int downRatio, const int layer_scale){

    Dtype p_xmin = predict_box.xmin();
    Dtype p_xmax = predict_box.xmax();
    Dtype p_ymin = predict_box.ymin();
    Dtype p_ymax = predict_box.ymax();
    Dtype p_area = (p_xmax - p_xmin) *(p_ymax - p_ymin);

    Dtype gt_xmin = gt_bbox.xmin();
    Dtype gt_xmax = gt_bbox.xmax();
    Dtype gt_ymin = gt_bbox.ymin();
    Dtype gt_ymax = gt_bbox.ymax();
    Dtype gt_area = (gt_xmax - gt_xmin) *(gt_ymax - gt_ymin);

    Dtype iou_xmin = std::max(p_xmin, gt_xmin), iou_xmax = std::min(p_xmax, gt_xmax);
    Dtype iou_ymin = std::max(p_ymin, gt_ymin), iou_ymax = std::min(p_ymax, gt_ymax);
    Dtype iou_height = (iou_ymax - iou_ymin);
    Dtype iou_width = (iou_xmax - iou_xmin);
    Dtype iou_area = iou_height * iou_width;
    Dtype Union = p_area + gt_area - iou_area;
    Dtype Iou = Dtype(iou_area / Union);

    Dtype c_xmin = std::min(p_xmin, gt_xmin), c_xmax = std::max(p_xmax, gt_xmax);
    Dtype c_ymin = std::min(p_ymin, gt_ymin), c_ymax = std::max(p_ymax, gt_ymax);
    Dtype C = (c_xmax - c_xmin) * (c_ymax - c_ymin);

    Dtype GIou = Iou - Dtype((C - Union) / C);

    // cal diff float IoU = I / U;
    // Partial Derivatives, derivatives
    Dtype dp_aera_wrt_xmin = -1 * (p_ymax - p_ymin);
    Dtype dp_aera_wrt_xmax = (p_ymax - p_ymin);
    Dtype dp_aera_wrt_ymin = -1 * (p_xmax - p_xmin);
    Dtype dp_aera_wrt_ymax = (p_xmax - p_xmin);

    // gradient of I min/max in IoU calc (prediction)
    Dtype dI_wrt_xmin = p_xmin > gt_xmin ? (-1 * iou_height) : 0;
    Dtype dI_wrt_xmax = p_xmax < gt_xmax ? iou_height : 0;
    Dtype dI_wrt_ymin = p_ymin > gt_ymin ? (-1 * iou_width) : 0;
    Dtype dI_wrt_ymax = p_ymax < gt_ymax ? iou_width : 0;

    // derivative of U with regard to x
    Dtype dU_wrt_ymin = dp_aera_wrt_ymin - dI_wrt_ymin;
    Dtype dU_wrt_ymax = dp_aera_wrt_ymax - dI_wrt_ymax;
    Dtype dU_wrt_xmin = dp_aera_wrt_xmin - dI_wrt_xmin;
    Dtype dU_wrt_xmax = dp_aera_wrt_xmax - dI_wrt_xmax;
    // gradient of C min/max in IoU calc (prediction)
    Dtype dC_wrt_ymin = p_ymin < gt_ymin ? (-1 * (c_xmax - c_xmin)) : 0;
    Dtype dC_wrt_ymax = p_ymax > gt_ymax ? (c_xmax - c_xmin) : 0;
    Dtype dC_wrt_xmin = p_xmin < gt_xmin ? (-1 * (c_ymax - c_ymin)) : 0;
    Dtype dC_wrt_xmax = p_xmax > gt_ymax ? (c_ymax - c_ymin) : 0;

    Dtype p_dt = Dtype(0.);
    Dtype p_db = Dtype(0.);
    Dtype p_dl = Dtype(0.);
    Dtype p_dr = Dtype(0.);
    if (Union > 0) {
      p_dt = ((Union * dI_wrt_ymin) - (iou_area * dU_wrt_ymin)) / (Union * Union);
      p_db = ((Union * dI_wrt_ymax) - (iou_area * dU_wrt_ymax)) / (Union * Union);
      p_dl = ((Union * dI_wrt_xmin) - (iou_area * dU_wrt_xmin)) / (Union * Union);
      p_dr = ((Union * dI_wrt_xmax) - (iou_area * dU_wrt_xmax)) / (Union * Union);
    }
    if (C > 0) {
        // apply "C" term from gIOU
        p_dt += ((C * dU_wrt_ymin) - (Union * dC_wrt_ymin)) / (C * C);
        p_db += ((C * dU_wrt_ymax) - (Union * dC_wrt_ymax)) / (C * C);
        p_dl += ((C * dU_wrt_xmin) - (Union * dC_wrt_xmin)) / (C * C);
        p_dr += ((C * dU_wrt_xmax) - (Union * dC_wrt_xmax)) / (C * C);
    }

    Dtype di_y1 = p_ymin < p_ymax ? p_dt : p_db;
    Dtype di_y2 = p_ymin < p_ymax ? p_db : p_dt;
    Dtype di_x1 = p_xmin < p_xmax ? p_dl : p_dr;
    Dtype di_x2 = p_xmin < p_xmax ? p_dr : p_dl;

    *diff_x1 = di_x1 * (-1) * Dtype(anchor_scale / (2 * downRatio * layer_scale));
    *diff_y1 = di_y1 * (-1) * Dtype(anchor_scale / (2 * downRatio * layer_scale));
    *diff_x2 = di_x2 * (-1) * Dtype(anchor_scale / (2 * downRatio * layer_scale));
    *diff_y2 = di_y2 * (-1) * Dtype(anchor_scale / (2 * downRatio * layer_scale));
    return (1 - GIou);
}

template float GIoULoss(NormalizedBBox predict_box, NormalizedBBox gt_bbox, float* diff_x1, 
                float* diff_x2, float* diff_y1, float* diff_y2, const int anchor_scale,
                const int downRatio, const int layer_scale);
template double GIoULoss(NormalizedBBox predict_box, NormalizedBBox gt_bbox, double* diff_x1, 
                double* diff_x2, double* diff_y1, double* diff_y2, const int anchor_scale,
                const int downRatio, const int layer_scale);


template <typename Dtype>
Dtype DIoULoss(NormalizedBBox predict_box, NormalizedBBox gt_bbox, Dtype* diff_x1, 
                Dtype* diff_x2, Dtype* diff_y1, Dtype* diff_y2){
    NOT_IMPLEMENTED;
}
}  // namespace caffe
