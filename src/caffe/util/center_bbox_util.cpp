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
#include "caffe/util/center_bbox_util.hpp"

int count_gt = 0;
int count_one = 0;
int count_zero = 0;
int count_no_zero = 0;

namespace caffe {
template<typename Dtype>
Dtype gaussian_radius(const Dtype heatmap_width, const Dtype heatmap_height, const Dtype min_overlap){

  Dtype a1  = Dtype(1.0);
  Dtype b1  = (heatmap_width + heatmap_height);
  Dtype c1  = heatmap_width * heatmap_height * (1 - min_overlap) / (1 + min_overlap);
  Dtype sq1 = std::sqrt(b1 * b1 - 4 * a1 * c1);
  Dtype r1  = Dtype(b1 + sq1) / 2;

  Dtype a2  = Dtype(4.0);
  Dtype b2  = 2 * (heatmap_height + heatmap_width);
  Dtype c2  = (1 - min_overlap) * heatmap_width * heatmap_height;
  Dtype sq2 = std::sqrt(b2 * b2 - 4 * a2 * c2);
  Dtype r2  = Dtype(b2 + sq2) / 2;

  Dtype a3  = Dtype(4 * min_overlap);
  Dtype b3  = -2 * min_overlap * (heatmap_height + heatmap_width);
  Dtype c3  = (min_overlap - 1) * heatmap_width * heatmap_height;
  Dtype sq3 = std::sqrt(b3 * b3 - 4 * a3 * c3);
  Dtype r3  = Dtype(b3 + sq3) / 2;
  return std::min(std::min(r1, r2), r3);
}

template float gaussian_radius(const float heatmap_width, const float heatmap_height, const float min_overlap);
template double gaussian_radius(const double heatmap_width, const double heatmap_height, const double min_overlap);

template <typename Dtype>
void EncodeCenteGroundTruthAndPredictions(const Dtype* loc_data, const int output_width, 
                                const int output_height, bool share_location, 
                                Dtype* pred_data, const int num_channels,
                                Dtype* gt_data, std::map<int, vector<NormalizedBBox> > all_gt_bboxes){
  std::map<int, vector<NormalizedBBox> > ::iterator iter;
  int count = 0;
  for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
    int batch_id = iter->first;
    vector<NormalizedBBox> gt_bboxes = iter->second;
    for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
      const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
      const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
      const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
      const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
      Dtype center_x = (xmin + xmax) / 2;
      Dtype center_y = (ymin + ymax) / 2;
      int inter_center_x = static_cast<int> (center_x);
      int inter_center_y = static_cast<int> (center_y);
      Dtype diff_x = center_x - inter_center_x;
      Dtype diff_y = center_y - inter_center_y;
      Dtype width = gt_bboxes[ii].xmax() - gt_bboxes[ii].xmin();
      Dtype height = gt_bboxes[ii].ymax() - gt_bboxes[ii].ymin();
      int dimScale = output_height * output_width;
      int x_loc_index = batch_id * num_channels * dimScale 
                                + inter_center_y * output_width + inter_center_x;
      int y_loc_index = batch_id * num_channels * dimScale 
                                + dimScale
                                + inter_center_y * output_width + inter_center_x;
      int width_loc_index = batch_id * num_channels * dimScale
                                + 2 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      int height_loc_index = batch_id * num_channels * dimScale 
                                + 3 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      gt_data[count * 4 + 0] = diff_x;
      gt_data[count * 4 + 1] = diff_y;
      gt_data[count * 4 + 2] = Dtype(width);
      gt_data[count * 4 + 3] = Dtype(height );
      pred_data[count * 4 + 0] = loc_data[x_loc_index];
      pred_data[count * 4 + 1] = loc_data[y_loc_index];
      pred_data[count * 4 + 2] = loc_data[width_loc_index];
      pred_data[count * 4 + 3] = loc_data[height_loc_index];
      count++;
      #if 0
      LOG(INFO)<<"diff_x: "<<diff_x<<", diff_y: "<<diff_y<<", width: "<<width  <<", height: "<<height;
      #endif
    }
  }
}
template void EncodeCenteGroundTruthAndPredictions(const float* loc_data, const int output_width, 
                                const int output_height, bool share_location, 
                                float* pred_data, const int num_channels,
                                float* gt_data, std::map<int, vector<NormalizedBBox> > all_gt_bboxes);
template void EncodeCenteGroundTruthAndPredictions(const double* loc_data, const int output_width, 
                                const int output_height, bool share_location, 
                                double* pred_data, const int num_channels,
                                double* gt_data, std::map<int, vector<NormalizedBBox> > all_gt_bboxes);                              

template <typename Dtype>
void CopyDiffToBottom(const Dtype* pre_diff, const int output_width, 
                                const int output_height, 
                                bool share_location, Dtype* bottom_diff, const int num_channels,
                                std::map<int, vector<NormalizedBBox> > all_gt_bboxes){
  std::map<int, vector<NormalizedBBox> > ::iterator iter;
  int count = 0;
  for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
    int batch_id = iter->first;
    vector<NormalizedBBox> gt_bboxes = iter->second;
    for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
      const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
      const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
      const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
      const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
      Dtype center_x = (xmin + xmax) / 2;
      Dtype center_y = (ymin + ymax) / 2;
      int inter_center_x = static_cast<int> (center_x);
      int inter_center_y = static_cast<int> (center_y);
      int dimScale = output_height * output_width;
      int x_loc_index = batch_id * num_channels * dimScale 
                                + inter_center_y * output_width + inter_center_x;
      int y_loc_index = batch_id * num_channels * dimScale 
                                + dimScale
                                + inter_center_y * output_width + inter_center_x;
      int width_loc_index = batch_id * num_channels * dimScale 
                                + 2 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      int height_loc_index = batch_id * num_channels * dimScale
                                + 3 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      bottom_diff[x_loc_index] = pre_diff[count * 4 + 0];
      bottom_diff[y_loc_index] = pre_diff[count * 4 + 1];
      bottom_diff[width_loc_index] = pre_diff[count * 4 + 2];
      bottom_diff[height_loc_index] = pre_diff[count * 4 + 3];
      count++;
    }
  }
}
template void CopyDiffToBottom(const float* pre_diff, const int output_width, 
                                const int output_height, 
                                bool share_location, float* bottom_diff, const int num_channels,
                                std::map<int, vector<NormalizedBBox> > all_gt_bboxes);
template void CopyDiffToBottom(const double* pre_diff, const int output_width, 
                                const int output_height, 
                                bool share_location, double* bottom_diff, const int num_channels,
                                std::map<int, vector<NormalizedBBox> > all_gt_bboxes);

template <typename Dtype>
void _nms_heatmap(const Dtype* conf_data, Dtype* keep_max_data, const int output_height
                  , const int output_width, const int channels, const int num_batch){
    int dim = num_batch * channels * output_height * output_width;
    for(int ii = 0; ii < dim; ii++){
      keep_max_data[ii] = (keep_max_data[ii] == conf_data[ii]) ? keep_max_data[ii] : Dtype(0.);
    }
}
template void _nms_heatmap(const float* conf_data, float* keep_max_data, const int output_height
                  , const int output_width, const int channels, const int num_batch);
template void _nms_heatmap(const double* conf_data, double* keep_max_data, const int output_height
                  , const int output_width, const int channels, const int num_batch);

template <typename Dtype>
void get_topK(const Dtype* keep_max_data, const Dtype* loc_data, const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo> >* results
                  , const int loc_channels){
  for(int i = 0; i < num_batch; i++){
    int dim = classes * output_width * output_height;
    std::vector<CenterNetInfo> batch_result;
    batch_result.clear();
    for(int c = 0 ; c < classes; c++){
      int dimScale = output_width * output_height;
      for(int h = 0; h < output_height; h++){
        for(int w = 0; w < output_width; w++){
          int index = i * dim + c * dimScale + h * output_width + w;
          if(keep_max_data[index] != 0){
            int x_loc_index = i * loc_channels * dimScale + h * output_width + w;
            int y_loc_index = i * loc_channels * dimScale + dimScale + h * output_width + w;
            int width_loc_index = i * loc_channels * dimScale + 2 * dimScale + h * output_width + w;
            int height_loc_index = i * loc_channels * dimScale + 3 * dimScale + h * output_width + w;
            float center_x = w + loc_data[x_loc_index];
            float center_y = h + loc_data[y_loc_index];
            float width = loc_data[width_loc_index] * output_width;
            float height = loc_data[height_loc_index] * output_height;
            CenterNetInfo temp_result = {
              .class_id = c,
              .score = keep_max_data[index],
              .xmin = float((center_x - float(width / 2)) / output_width),
              .ymin = float((center_y - float(height / 2)) / output_height),
              .xmax = float((center_x + float(width / 2)) / output_width),
              .ymax = float((center_y + float(height / 2)) / output_height)
            };
            batch_result.push_back(temp_result);
          } 
        }
      }
    }
    if(batch_result.size() > 0){
      if(results->find(i) == results->end()){
        results->insert(std::make_pair(i, batch_result));
      }else{
      }
    }
  }
}
template  void get_topK(const float* keep_max_data, const float* loc_data, const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo> >* results
                  , const int loc_channels);
template void get_topK(const double* keep_max_data, const double* loc_data, const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo> >* results
                  , const int loc_channels);

#ifdef USE_OPENCV

cv::Mat gaussian2D(const int height, const int width, const float sigma){
  int half_width = (width - 1) / 2;
  int half_height = (height - 1) / 2;
  cv::Mat heatmap(cv::Size(width, height), CV_32FC1, cv::Scalar(0));
  for(int i = 0; i < heatmap.rows; i++){
    float *data = heatmap.ptr<float>(i);
    int x = i - half_width;
    for(int j = 0; j < heatmap.cols; j++){
      int y = j - half_height;
      data[j] = std::exp(-(x*x + y*y)/ (2* sigma * sigma));
      if(data[j] < 0)
        data[j] = 0;
    }
  }
  return heatmap;
}

void draw_umich_gaussian(cv::Mat heatmap, int center_x, int center_y, float radius, int k = 1){
  float diameter = 2 * radius + 1;
  cv::Mat gaussian = gaussian2D(int(diameter), int(diameter), float(diameter / 6));
  int height = heatmap.rows, width = heatmap.cols;
  int left = std::min(int(center_x), int(radius)), right = std::min(int(width - center_x), int(radius) + 1);
  int top = std::min(int(center_y), int(radius)), bottom = std::min(int(height - center_y), int(radius) + 1);
  if((left + right) > 0 && (top + bottom) > 0){
    cv::Mat masked_heatmap = heatmap(cv::Rect(int(center_x) -left, int(center_y) -top, (right + left), (bottom + top)));
    cv::Mat masked_gaussian = gaussian(cv::Rect(int(radius) - left, int(radius) - top, (right + left), (bottom + top)));
    for(int row = 0; row < (top + bottom); row++){
      float *masked_heatmap_data = masked_heatmap.ptr<float>(row);
      float *masked_gaussian_data = masked_gaussian.ptr<float>(row);
      for(int col = 0; col < (right + left); col++){
        masked_heatmap_data[col] = masked_heatmap_data[col] >= masked_gaussian_data[col] * k ? masked_heatmap_data[col]:
                                      masked_gaussian_data[col] * k;
      }
    }
  }
  #if 0
  LOG(INFO)<<"left + right: "<<left + right<<",top + bottom: "<<top + bottom; 
  for(int row = 0; row < height; row++){
    float *masked_heatmap_data = heatmap.ptr<float>(row);
    for(int col = 0; col < width; col++){
      if(masked_heatmap_data[col] == 1.f)
        LOG(INFO)<<"heatmap center_x: "<< col << ", heatmap center_y; "<< row;
    }
  }
  #endif
}

template <typename Dtype>
void transferCVMatToBlobData(cv::Mat heatmap, Dtype* buffer_heat){
  int width = heatmap.cols;
  int height = heatmap.rows;
  for(int row = 0; row < height; row++){
    float* data = heatmap.ptr<float>(row);
    for(int col = 0; col < width; col++){
      buffer_heat[row*width + col] = buffer_heat[row*width + col] > data[col] ? 
                                              buffer_heat[row*width + col] : data[col];
    }
  }
}
template void transferCVMatToBlobData(cv::Mat heatmap, float* buffer_heat);
template void transferCVMatToBlobData(cv::Mat heatmap, double* buffer_heat);


template <typename Dtype>
void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, Dtype* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height){
  std::map<int, vector<NormalizedBBox> > ::iterator iter;
  count_gt = 0;
  count_zero = 0;
  count_no_zero = 0;
  count_one = 0;
  for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
    int batch_id = iter->first;
    vector<NormalizedBBox> gt_bboxes = iter->second;
    for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
      cv::Mat heatmap(cv::Size(output_width, output_height), CV_32FC1, cv::Scalar(0));
      const int class_id = gt_bboxes[ii].label();
      Dtype *classid_heap = gt_heatmap + (batch_id * num_classes_ + (class_id - 1)) * output_width * output_height;
      const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
      const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
      const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
      const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
      const Dtype width = Dtype(xmax - xmin);
      const Dtype height = Dtype(ymax - ymin);
      Dtype radius = gaussian_radius(width, height, Dtype(0.7));
      radius = std::max(0, int(radius));
      int center_x = int( (xmin + xmax) / 2 );
      int center_y = int( (ymin + ymax) / 2 );
      #if 0
      LOG(INFO)<<"batch_id: "<<batch_id<<", class_id: "
                <<class_id<<", radius: "<<radius<<", center_x: "
                <<center_x<<", center_y: "<<center_y<<", output_height: "
                <<output_height<<", output_width: "<<output_width
                <<", bbox_width: "<<width<<", bbox_height: "<<height;
      #endif
      draw_umich_gaussian( heatmap, center_x, center_y, radius );
      transferCVMatToBlobData(heatmap, classid_heap);
      count_gt++;
    }
  }
  #if 1
  for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
    int batch_id = iter->first;
    vector<NormalizedBBox> gt_bboxes = iter->second;
    for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
      const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
      const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
      const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
      const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
      int center_x = int( (xmin + xmax) / 2 );
      int center_y = int( (ymin + ymax) / 2 );
      //LOG(INFO)<<"gt_bboxes center_x: "<< center_x << ", gt_bboxes center_y; "<< center_y;
    }
    for(int c = 0; c < num_classes_; c++){
      for(int h = 0 ; h < output_height; h++){
        for(int w = 0; w < output_width; w++){
          int index = batch_id * num_classes_ * output_height * output_width + c * output_height * output_width + h * output_width + w;  
          if(gt_heatmap[index] == 0.f)
            count_zero++;
          if(gt_heatmap[index] != 0.f)
            count_no_zero++;
          if(gt_heatmap[index] == 1.f){
            count_one++;
            //LOG(INFO)<<"heatmap center_x: "<< w << ", heatmap center_y; "<< h << ", value: "<<gt_heatmap[index];
          }
        }
      }
    }
  }
  //LOG(INFO)<<"count_no_zero: "<<count_no_zero<<", count_zero: "<<count_zero;
  CHECK_EQ(count_one, count_gt);
  #endif
}
template void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, float* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);
template void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, double* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);
#endif  // USE_OPENCV

}  // namespace caffe
