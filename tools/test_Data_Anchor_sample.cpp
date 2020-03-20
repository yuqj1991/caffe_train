#include <algorithm>
#include <csignal>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/bbox_util.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
using namespace cv;
using namespace std;

#define COMPAREMIN_T(a, b) (a >= b ? b : a)
#if 1
struct NormalizedBBox_S{
   float xmin;
   float ymin;
   float xmax;
   float ymax;
};

bool SatisfySampleConstraint_F(NormalizedBBox_S sample_bbox, 
                              std::vector<NormalizedBBox_S> object_bboxes,
                                    float min_coverage){
  bool found = false;
  NormalizedBBox test_bbox;
  test_bbox.set_xmin(sample_bbox.xmin);
  test_bbox.set_xmax(sample_bbox.xmax);
  test_bbox.set_ymin(sample_bbox.ymin);
  test_bbox.set_ymax(sample_bbox.ymax);
  for(unsigned ii = 0; ii < object_bboxes.size(); ii ++){
    NormalizedBBox gt_bbox;
    gt_bbox.set_xmin(object_bboxes[ii].xmin);
    gt_bbox.set_xmax(object_bboxes[ii].xmax);
    gt_bbox.set_ymin(object_bboxes[ii].ymin);
    gt_bbox.set_ymax(object_bboxes[ii].ymax);
    const float object_coverage = BBoxCoverage(gt_bbox, test_bbox);
    if(min_coverage < object_coverage)
      continue;
    found = true;
    if (found) {
      return true;
    }
  }
  return found;
}

std::vector<NormalizedBBox_S> ResizedCropSample(const cv::Mat& src_img, cv::Mat resized_img, float scale, 
                        std::vector<NormalizedBBox_S> src_gt_bboxes){
  const int img_width = src_img.cols;
  const int img_height = src_img.rows;

  // image data
  int Resized_img_Height = int(img_height * scale);
  int Resized_img_Width = int(img_width * scale);
  cv::resize(src_img, resized_img, cv::Size(Resized_img_Width, Resized_img_Height), 0, 0,
                cv::INTER_CUBIC);
  std::vector<NormalizedBBox_S> Resized_gt_bboxes;
  for(unsigned ii = 0; ii < src_gt_bboxes.size(); ii++){
    float x_min = src_gt_bboxes[ii].xmin * img_width;
    float y_min = src_gt_bboxes[ii].ymin * img_height;
    float x_max = src_gt_bboxes[ii].xmax * img_width;
    float y_max = src_gt_bboxes[ii].ymax * img_height;
    x_min = std::max(0.f, x_min * Resized_img_Width / img_width);
    x_max = std::min(float(Resized_img_Width), x_max * Resized_img_Width / img_width);
    y_min = std::max(0.f, y_min * Resized_img_Height / img_height);
    y_max = std::min(float(Resized_img_Height), y_max * Resized_img_Height / img_height);
    NormalizedBBox_S Resized_bbox = {
      .xmin = x_min,
      .ymin = y_min,
      .xmax = x_max,
      .ymax = y_max
    };
    Resized_gt_bboxes.push_back(Resized_bbox);
  }
  return Resized_gt_bboxes;
}
void GenerateDataAnchorSample(cv::Mat img, 
                                std::vector<int>anchorScale,
                                const vector<NormalizedBBox_S>& object_bboxes,
                                int resized_height, int resized_width,
                                NormalizedBBox_S* sampled_bbox, cv::Mat resized_img,
                                bool do_resize){
  int img_height = img.rows;
  int img_width = img.cols;
  CHECK_GT(object_bboxes.size(), 0);
  int object_bbox_index = caffe_rng_rand() % object_bboxes.size();
  const float xmin = object_bboxes[object_bbox_index].xmin*img_width;
  const float xmax = object_bboxes[object_bbox_index].xmax*img_width;
  const float ymin = object_bboxes[object_bbox_index].ymin*img_height;
  const float ymax = object_bboxes[object_bbox_index].ymax*img_height;
  float bbox_width = xmax - xmin;
  float bbox_height = ymax - ymin;
  int range_size = 0, range_idx_size = 0, rng_random_index = 0; 
  float bbox_aera = bbox_height * bbox_width;
  float scaleChoose = 0.0f; 
  float min_resize_val = 0.f, max_resize_val = 0.f;
  for(int j = 0; j < anchorScale.size() -1; ++j){
    if(bbox_aera >= std::pow(anchorScale[j], 2) && bbox_aera < std::pow(anchorScale[j+1], 2)){
      range_size = j + 1;
      break;
    }
  }
  if(bbox_aera > std::pow(anchorScale[anchorScale.size() - 2], 2))
    range_size = anchorScale.size() - 2;
  if(range_size==0){
    range_idx_size = 0;
  }else{
    rng_random_index = caffe_rng_rand() % (range_size + 1);
    range_idx_size = rng_random_index % (range_size + 1);
  }
  if(range_idx_size == range_size){
    min_resize_val = anchorScale[range_idx_size] / 2;
    max_resize_val = COMPAREMIN_T((float)anchorScale[range_idx_size] * 2,
                                                  2*std::sqrt(bbox_aera)) ;
    caffe_rng_uniform(1, min_resize_val, max_resize_val, &scaleChoose);
  }else{
    min_resize_val = anchorScale[range_idx_size] / 2;
    max_resize_val = (float)anchorScale[range_idx_size] * 2;
    caffe_rng_uniform(1, min_resize_val, max_resize_val, &scaleChoose);
  }
   
  float width_offset_org = 0.0f, height_offset_org = 0.0f;
  float w_off = 0.0f, h_off = 0.0f, w_end = 0.0f, h_end = 0.0f;
  if(do_resize){
    float scale = (float) scaleChoose / bbox_width;
    ResizedCropSample(img, resized_img, scale, object_bboxes);
    int Resized_ori_Height = int(scale * img_height);
    int Resized_ori_Width = int(scale * img_width);
    int Resized_bbox_width = int(scale * bbox_width);
    int Resized_bbox_height = int(scale * bbox_height);
    const float Resized_xmin = object_bboxes[object_bbox_index].xmin*Resized_ori_Width;
    const float Resized_ymin = object_bboxes[object_bbox_index].ymin*Resized_ori_Height;
    if(resized_width < std::max(Resized_ori_Height, Resized_ori_Width)){
      if(Resized_bbox_width <= resized_width){
        if(Resized_bbox_width == resized_width){
          width_offset_org = xmin;
        }else{
          caffe_rng_uniform(1, Resized_bbox_width + Resized_xmin - resized_width, Resized_xmin, &width_offset_org );
        }
      }else{
        caffe_rng_uniform(1, Resized_xmin, Resized_bbox_width + Resized_xmin - resized_width, &width_offset_org);
      }
      if(Resized_bbox_height <= resized_height){
        if(Resized_bbox_height == resized_height){
          height_offset_org = ymin;
        }else{
          caffe_rng_uniform(1, Resized_ymin + Resized_bbox_height - resized_height, Resized_ymin, &height_offset_org);
        }
      }else{
        caffe_rng_uniform(1, Resized_ymin, Resized_ymin + Resized_bbox_height - resized_height, &height_offset_org);
      }
    }else{
      caffe_rng_uniform(1, float(Resized_ori_Width-resized_width), 0.0f, &width_offset_org);
      caffe_rng_uniform(1, float(Resized_ori_Height-resized_height), 0.0f, &height_offset_org);
    }
    int width_offset_ = std::floor(width_offset_org);
    int height_offset_ = std::floor(height_offset_org);
    w_off = (float) width_offset_ / Resized_ori_Width;
    h_off = (float) height_offset_ / Resized_ori_Height;
    w_end = w_off + float(resized_width / Resized_ori_Width);
    h_end = h_off + float(resized_height / Resized_ori_Height);
  }
  sampled_bbox->xmin = w_off;
  sampled_bbox->xmin = h_off;
  sampled_bbox->xmax = w_end;
  sampled_bbox->ymax = h_end;
}

void GenerateBatchDataAnchorSamples(const cv::Mat src_img, vector<NormalizedBBox_S> object_bboxes, 
                                const vector<std::vector<int> >& data_anchor_samplers,
                                int resized_height, int resized_width, 
                                NormalizedBBox_S* sampled_bbox, cv::Mat resized_img, 
                                bool do_resize, int max_sample) {
  CHECK_EQ(data_anchor_samplers.size(), 1);
  for (int i = 0; i < data_anchor_samplers.size(); ++i) {
    int found = 0;
    for (int j = 0; j < 50; ++j) {
      if (found >= max_sample) {
        break;
      }
      NormalizedBBox_S temp_sampled_bbox;
      GenerateDataAnchorSample(src_img, data_anchor_samplers[i], object_bboxes, resized_height, 
                              resized_width, &temp_sampled_bbox, resized_img, do_resize);
      if (SatisfySampleConstraint_F(temp_sampled_bbox, object_bboxes, 0.85)){
        found++;
        *sampled_bbox = temp_sampled_bbox;
      }
    }
    if(found == 0){
      src_img.copyTo(resized_img);
      sampled_bbox->xmin = 0.f;
      sampled_bbox->ymin = 0.f;
      sampled_bbox->xmax = 1.f;
      sampled_bbox->ymax = 1.f;
    }
  }
}



void Crop_img(){

}
                              
#endif

int main(){
  int loop_time = 12;
  int batch_size = 32;
  std::vector<string> img_filenames;
  std::vector<std::vector<NormalizedBBox_S> >all_gt_bboxes;
  // 读文件，文件里面存着真实值，包括图像文件， 和真是坐标值文件

  // 循环操作
  for(int ii = 0; ii < loop_time; ii++){
    for(int jj = 0; jj < batch_size; jj++){
      ;
    }
  }

	return 1;
}
