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
#include "caffe/caffe.hpp"
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
#define COMPAREMAX_T(a, b) (a >= b ? a : b)

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

std::vector<NormalizedBBox_S> ResizedCropSample(const cv::Mat& src_img, cv::Mat *resized_img, float scale, 
                        std::vector<NormalizedBBox_S> src_gt_bboxes){
  const int img_width = src_img.cols;
  const int img_height = src_img.rows;

  // image data
  int Resized_img_Height = int(img_height * scale);
  int Resized_img_Width = int(img_width * scale);
  cv::resize(src_img, *resized_img, cv::Size(Resized_img_Width, Resized_img_Height), 0, 0, cv::INTER_CUBIC);
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
      .xmin = x_min / Resized_img_Width,
      .ymin = y_min / Resized_img_Height,
      .xmax = x_max / Resized_img_Width,
      .ymax = y_max / Resized_img_Height
    };
    Resized_gt_bboxes.push_back(Resized_bbox);
  }
  return Resized_gt_bboxes;
}
void GenerateDataAnchorSample(cv::Mat img, 
                                std::vector<int>anchorScale,
                                const vector<NormalizedBBox_S>& object_bboxes,
                                int resized_height, int resized_width,
                                NormalizedBBox_S* sampled_bbox, cv::Mat *resized_img,
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
    int Resized_ori_Height = int(scale * img_height);
    int Resized_ori_Width = int(scale * img_width);
    int Resized_bbox_width = int(scale * bbox_width);
    int Resized_bbox_height = int(scale * bbox_height);
    std::cout<<"srcOri img_width: "<<img_width<<", srcOri img_height: "<<img_height
              <<", srcOri bbox_width: "<<bbox_width<<", srcOri bbox_height: "<<bbox_height
              <<std::endl;
    std::cout <<"scaleChoose: "<<scaleChoose<<", Resized bbox_width: "<<Resized_bbox_width<<", Resized_bbox_height: "<<Resized_bbox_height
              <<", Resized_ori_Width: "<<Resized_ori_Width<<", Resized_ori_Height: "<<Resized_ori_Height << ", "<<", scale: "<<scale
              <<std::endl;
    std::vector<NormalizedBBox_S>resiezed_gt_bboxes = ResizedCropSample(img, resized_img, scale, object_bboxes);
    CHECK_EQ(Resized_ori_Height, resized_img->rows);
    CHECK_EQ(Resized_ori_Width, resized_img->cols);
    const float Resized_xmin = object_bboxes[object_bbox_index].xmin * Resized_ori_Width;
    const float Resized_ymin = object_bboxes[object_bbox_index].ymin * Resized_ori_Height;
    std::cout <<"Resized_bbox_width + Resized_xmin - resized_width: "<<Resized_bbox_width + Resized_xmin - resized_width<<", Resized_xmin: "<<Resized_xmin<<std::endl;
    std::cout <<"Resized_ymin + Resized_bbox_height - resized_height: "<<Resized_ymin + Resized_bbox_height - resized_height<<", Resized_ymin: "<<Resized_ymin<<std::endl;
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
    w_end = w_off + (float)resized_width / Resized_ori_Width;
    h_end = h_off + (float)resized_height / Resized_ori_Height;
    std::cout << width_offset_ << ", "<<height_offset_<<", "<<resized_width<<", "<<resized_height<<std::endl;
    std::cout << w_off << ", "<<h_off<<", "<<w_end<<", "<<h_end<<std::endl;
  }
  sampled_bbox->xmin = w_off;
  sampled_bbox->ymin = h_off;
  sampled_bbox->xmax = w_end;
  sampled_bbox->ymax = h_end;
}

void GenerateBatchDataAnchorSamples(const cv::Mat src_img, vector<NormalizedBBox_S> object_bboxes, 
                                const vector<std::vector<int> >& data_anchor_samplers,
                                int resized_height, int resized_width, 
                                NormalizedBBox_S* sampled_bbox, cv::Mat* resized_img, 
                                bool do_resize, int max_sample) {
  CHECK_EQ(data_anchor_samplers.size(), 1);
  for (int i = 0; i < data_anchor_samplers.size(); ++i) {
    int found = 0;
    for (int j = 0; j < 50; ++j) {
      if (found >= max_sample) {
        break;
      }
      NormalizedBBox_S temp_sampled_bbox;
      cv::Mat temp_resized_img;
      GenerateDataAnchorSample(src_img, data_anchor_samplers[i], object_bboxes, resized_height, 
                              resized_width, &temp_sampled_bbox, &temp_resized_img, do_resize);
      if (SatisfySampleConstraint_F(temp_sampled_bbox, object_bboxes, 0.85)){
        found++;
        *sampled_bbox = temp_sampled_bbox;
        temp_resized_img.copyTo(*resized_img);
      }
    }
    if(found == 0){
      src_img.copyTo(*resized_img);
      sampled_bbox->xmin = 0.f;
      sampled_bbox->ymin = 0.f;
      sampled_bbox->xmax = 1.f;
      sampled_bbox->ymax = 1.f;
    }
  }
  std::cout <<"sampled_bbox: xmin: "<<sampled_bbox->xmin <<", xmax: "<<sampled_bbox->xmax
                  <<", ymin: "<<sampled_bbox->ymin<<", ymax: "<<sampled_bbox->ymax<<std::endl;
  std::cout<<"Resized img Width: "<<resized_img->cols<<", Resized img Height: "<<resized_img->rows <<std::endl;
}

void CropImageData_Anchor_T(const cv::Mat& img,
									const NormalizedBBox& bbox, cv::Mat* crop_img) {
	int img_height = img.rows;
	int img_width = img.cols;
	float xmin = bbox.xmin() * img_width;
	float ymin = bbox.ymin() * img_height;
	float xmax = bbox.xmax() * img_width;
	float ymax = bbox.ymax() * img_height;
	
	float w_off = xmin, h_off = ymin, width = xmax - xmin, height = ymax - ymin;

	float cross_xmin = std::min(std::max(0.f, w_off), float(img_width - 1));
	float cross_ymin = std::min(std::max(0.f, h_off), float(img_height - 1)); 
	float cross_xmax = std::min(std::max(0.f, w_off + width), float(img_width - 1));
	float cross_ymax = std::min(std::max(0.f, h_off + height), float(img_height - 1));
	int cross_width = static_cast<int>(cross_xmax - cross_xmin);
	int	cross_height = static_cast<int>(cross_ymax - cross_ymin);

	float roi_xmin = w_off >= 0 ? 0 : std::fabs(w_off);
	float roi_ymin = h_off >= 0 ? 0 : std::fabs(h_off);
	int roi_width = cross_width;
	int roi_height = cross_height;

	int roi_x1 = static_cast<int>(roi_xmin);
	int roi_y1 = static_cast<int>(roi_ymin);
	int cross_x1 = static_cast<int>(cross_xmin);
	int cross_y1 = static_cast<int>(cross_ymin);
	crop_img->create(int(height), int(width), CV_8UC3);
	crop_img->setTo(cv::Scalar(0));
	cv::Rect bbox_cross(cross_x1, cross_y1, cross_width, cross_height);
	cv::Rect bbox_roi(roi_x1, roi_y1, roi_width, roi_height);
	img(bbox_cross).copyTo((*crop_img)(bbox_roi));
}

void transformGroundTruth(std::vector<NormalizedBBox_S> gt_bboxes,
                        bool do_resize, const int Resized_Height, const int Resized_Width,
                        const int img_Height, const int img_Width, 
                        const NormalizedBBox_S sampled_bbox,
                        std::vector<NormalizedBBox_S > *trans_gt_bboxes){
  for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
    NormalizedBBox_S bbox;
    if(do_resize){
      float x_min = gt_bboxes[ii].xmin * img_Width;
      float y_min = gt_bboxes[ii].ymin * img_Height;
      float x_max = gt_bboxes[ii].xmax * img_Width;
      float y_max = gt_bboxes[ii].ymax * img_Height;
      x_min = std::max(0.f, x_min * Resized_Width / img_Width);
      x_max = std::min(float(Resized_Width), x_max * Resized_Width / img_Width);
      y_min = std::max(0.f, y_min * Resized_Height / img_Height);
      y_max = std::min(float(Resized_Height), y_max * Resized_Height / img_Height);
      NormalizedBBox_S Resized_bbox = {
        .xmin = x_min / Resized_Width,
        .ymin = y_min / Resized_Height,
        .xmax = x_max / Resized_Width,
        .ymax = y_max / Resized_Height
      };
      bbox = Resized_bbox;
    }else{
      bbox = gt_bboxes[ii];
    }
    NormalizedBBox proj_bbox, crop_bbox, resized_bbox;
    resized_bbox.set_xmin(bbox.xmin);
    resized_bbox.set_xmax(bbox.xmax);
    resized_bbox.set_ymin(bbox.ymin);
    resized_bbox.set_ymax(bbox.ymax);

    crop_bbox.set_xmin(sampled_bbox.xmin);
    crop_bbox.set_xmax(sampled_bbox.xmax);
    crop_bbox.set_ymin(sampled_bbox.ymin);
    crop_bbox.set_ymax(sampled_bbox.ymax);
    if (ProjectBBox(crop_bbox, resized_bbox, &proj_bbox)) {
      bbox = {
        .xmin = proj_bbox.xmin(),
        .ymin = proj_bbox.ymin(),
        .xmax = proj_bbox.xmax(),
        .ymax = proj_bbox.ymax()
      };
      trans_gt_bboxes->push_back(bbox);
    }
  }
  std::cout << "TRANSFORM SUCCESSFULLY!"<<std::endl;
}


void Crop_Image_F(const cv::Mat src_img, cv::Mat * crop_img, 
              const NormalizedBBox_S sampled_bbox, 
              std::vector<NormalizedBBox_S> gt_bboxes, 
              std::vector<NormalizedBBox_S> *trans_gt_bboxes){
  NormalizedBBox bbox;
  bbox.set_xmin(sampled_bbox.xmin);
  bbox.set_xmax(sampled_bbox.xmax);
  bbox.set_ymin(sampled_bbox.ymin);
  bbox.set_ymax(sampled_bbox.ymax);
  CropImageData_Anchor_T(src_img, bbox, crop_img);
  int img_Height = src_img.rows;
  int img_Width = src_img.cols;
  bool do_resize = false;
  transformGroundTruth(gt_bboxes, do_resize, 
                        0, 0, img_Height, img_Width, 
                        sampled_bbox, trans_gt_bboxes);
}

void GenerateLffdSample_T(const cv::Mat& src_Img, std::vector<NormalizedBBox_S> object_bboxes, 
                        int resized_height, int resized_width,
                        NormalizedBBox_S* sampled_bbox, 
                        std::vector<int> bbox_small_size_list,
                        std::vector<int> bbox_large_size_list,
                        std::vector<int> anchorStride, 
                        cv::Mat* resized_Img, bool do_resize){
  CHECK_EQ(bbox_large_size_list.size(), bbox_small_size_list.size());
  int num_output_scale = bbox_small_size_list.size();
  int img_height = src_Img.rows;
  int img_width = src_Img.cols;
  CHECK_GT(object_bboxes.size(), 0);
  int object_bbox_index = caffe_rng_rand() % object_bboxes.size();
  const float xmin = object_bboxes[object_bbox_index].xmin*img_width;
  const float xmax = object_bboxes[object_bbox_index].xmax*img_width;
  const float ymin = object_bboxes[object_bbox_index].ymin*img_height;
  const float ymax = object_bboxes[object_bbox_index].ymax*img_height;
  float bbox_width = xmax - xmin;
  float bbox_height = ymax - ymin;
  float longer_side = COMPAREMAX_T(bbox_height, bbox_width);
  int scaled_idx = 0, side_length = 0;
  if(longer_side <= bbox_small_size_list[0]){
    scaled_idx = 0;
  }else if(longer_side <= bbox_small_size_list[2]){
    scaled_idx = caffe_rng_rand() % 3;
  }else if(longer_side >= bbox_small_size_list[num_output_scale - 1]){
    scaled_idx = num_output_scale - 1;
  }else{
    for(int ii = 3; ii < num_output_scale - 1; ii++){
      if(longer_side >= bbox_small_size_list[ii] && longer_side < bbox_small_size_list[ii + 1])
        scaled_idx = ii;
    }
  }
  if(scaled_idx == (num_output_scale - 1)){
    side_length = bbox_large_size_list[num_output_scale - 1] 
                    + caffe_rng_rand() % (static_cast<int>(bbox_large_size_list[num_output_scale - 1] * 0.5));
  }else{
    side_length = bbox_small_size_list[scaled_idx] 
                    + caffe_rng_rand() % (bbox_large_size_list[scaled_idx] - 
                                          bbox_small_size_list[scaled_idx]);
  }
  if(do_resize){
    float scale = (float) side_length / bbox_width;
    std::vector<NormalizedBBox_S> trans_gt_bboxes = ResizedCropSample(src_Img, resized_Img, scale, object_bboxes);
    int Resized_ori_Height = int(scale * img_height);
    int Resized_ori_Width = int(scale * img_width);
    NormalizedBBox_S target_bbox = object_bboxes[object_bbox_index];
    float resized_xmin = target_bbox.xmin * Resized_ori_Width;
    float resized_xmax = target_bbox.xmax * Resized_ori_Width;
    float resized_ymin = target_bbox.ymin * Resized_ori_Height;
    float resized_ymax = target_bbox.ymax * Resized_ori_Height;
    float vibration_length = float(anchorStride[scaled_idx]);
    float offset_x = 0.f, offset_y = 0.f;
    caffe_rng_uniform(1, -vibration_length, vibration_length, &offset_x);
    caffe_rng_uniform(1, -vibration_length, vibration_length, &offset_y);
    float width_offset_ = (resized_xmin + resized_xmax) / 2 + offset_x - resized_width / 2;
    float height_offset_ = (resized_ymin + resized_ymax) / 2 + offset_y - resized_height / 2;
    float width_end_ = (resized_xmin + resized_xmax) / 2 + offset_x + resized_width / 2;
    float height_end_ = (resized_ymin + resized_ymax) / 2 + offset_y + resized_height / 2;
    float w_off = (float) width_offset_ / Resized_ori_Width;
    float h_off = (float) height_offset_ / Resized_ori_Height;
    float w_end = (float) width_end_ / Resized_ori_Width;
    float h_end = (float) height_end_ / Resized_ori_Height;
    sampled_bbox->xmin = (w_off);
    sampled_bbox->ymin = (h_off);
    sampled_bbox->xmax = (w_end);
    sampled_bbox->ymax = (h_end);
  }
}

int main(int argc, char** argv){
  int loop_time = 12;
  int batch_size = 32;
  int Resized_Height = 640;
  int Resized_Width = 640;
  std::string srcTestfile = "/home/deepano/workspace/img_test.txt";
  std::string save_folder = "/home/deepano/workspace/anchorTestImage";
  std::vector<std::pair<string, string> > img_filenames;
  std::map<int, std::vector<NormalizedBBox_S> >all_gt_bboxes;
  // anchor samples policy
  // anchor 的大概范围16, 32, 64, 128, 256, 512
  std::vector<int> Anchors;
  for(unsigned nn = 0; nn < 6; nn++){
    Anchors.push_back(16 * std::pow(2, nn));
  }
  std::vector<std::vector<int> >anchorSamples;
  anchorSamples.push_back(Anchors);
  // 设计LFFD 方式的Crop裁剪方法
  // low gt_boxes_list: 10, 15, 20, 40, 70, 110, 250, 400
  // up gt_boxes_list: 15, 20, 40, 70, 110, 250, 400, 560
  // anchorStride_list: 4, 4, 8, 8, 16, 32, 32, 32
  int low_boxes_list[8] = { 10, 15, 20, 40, 70, 110, 250, 400};
  int up_boxes_list[8] = {15, 20, 40, 70, 110, 250, 400, 560};
  int anchorStride_list[8] = {4, 4, 8, 8, 16, 32, 32, 32};
  std::vector<int> low_gt_boxes_list, up_gt_boxes_list, anchor_stride_list;
  for (size_t i = 0; i < 8; i++)
  {
    low_gt_boxes_list.push_back(low_boxes_list[i]);
    up_gt_boxes_list.push_back(up_boxes_list[i]);
    anchor_stride_list.push_back(anchorStride_list[i]);
  }
  // 读文件，文件里面存着真实值，包括图像文件， 和真是坐标值文件
  // 随机裁剪，生成再Resize到相对应大小的（640， 640）
  std::ifstream infile(srcTestfile.c_str());
  string line;
  size_t pos;
  std::stringstream sstr ;
  while(std::getline(infile, line)){
    pos = line.find_last_of(' ');
    std::string label_file = line.substr(pos+1);
    std::string img_file = line.substr(0, pos);
    img_filenames.push_back(std::make_pair(img_file, label_file));
  }
  infile.close();
  int numSamples = img_filenames.size();
  for(int ii = 0; ii < numSamples; ii++){
    cv::Mat srcImg = cv::imread(img_filenames[ii].first);
    int img_Height = srcImg.rows;
    int img_Width = srcImg.cols;
    std::string label_file = img_filenames[ii].second;
    infile.open(label_file.c_str());
    float xmin, ymin, width, height, blur, occur;
    while(std::getline(infile, line)){
      sstr << line;
      sstr >> xmin >> ymin >> width >> height >>blur >> occur;
      //std::cout<< xmin<<", " << ymin <<", "<< width <<", "<< height<<std::endl;
      float xmax = xmin + width;
      float ymax = ymin + height;
      NormalizedBBox_S label_bbox = {
        .xmin = xmin / img_Width,
        .ymin = ymin / img_Height,
        .xmax = xmax / img_Width,
        .ymax = ymax / img_Height
      };
      all_gt_bboxes[ii].push_back(label_bbox);
      sstr.clear();
    }
    infile.close();
  }
  // 循环操作
  for(int ii = 0; ii < loop_time; ii++){
    for(int jj = 0; jj < batch_size; jj++){
      int rand_idx = caffe_rng_rand() % numSamples;
      int pos_name = img_filenames[rand_idx].first.find_last_of("/");
      std::string img_name = img_filenames[rand_idx].first.substr(pos_name);
      int pos_suffix = img_name.find_last_of(".");
      std::string prefix_imgName = img_name.substr(0, pos_suffix);     
      NormalizedBBox_S anchorSampled_bbox;
      std::vector<NormalizedBBox_S> transfor_gt_bboxes;
      cv::Mat Resized_img, cropImage;
      cv::Mat srcImg = cv::imread(img_filenames[rand_idx].first);
      std::cout<<"file: "<<img_filenames[rand_idx].first<<std::endl;
      #if 0
      GenerateBatchDataAnchorSamples(srcImg, all_gt_bboxes.find(rand_idx)->second, anchorSamples, Resized_Height, 
                                      Resized_Width, &anchorSampled_bbox, &Resized_img, true, 20);
      #else
      GenerateLffdSample_T(srcImg, all_gt_bboxes.find(rand_idx)->second, Resized_Height, Resized_Width, &anchorSampled_bbox,
                                      low_gt_boxes_list, up_gt_boxes_list, anchor_stride_list, &Resized_img, true);
      #endif
      std::cout<<"SAMPEL SUCCESSFULLY"<<std::endl;
      //cv::imshow("Gaussian", Resized_img);
	    //cv::waitKey(0);
      Crop_Image_F(Resized_img, &cropImage, anchorSampled_bbox, all_gt_bboxes.find(rand_idx)->second, &transfor_gt_bboxes);
      std::cout<<"CROP IMAGE SAMPEL SUCCESSFULLY"<<std::endl;
      std::string saved_img_name = save_folder + "/" + prefix_imgName + "_" + to_string(ii) + "_" + to_string(jj) +".jpg";
      std::cout << "gt transfor_gt_bboxes size: "<< transfor_gt_bboxes.size()<<std::endl;
      int Crop_Height = cropImage.rows;
      int Crop_Width = cropImage.cols;
      for(unsigned nn = 0; nn < transfor_gt_bboxes.size(); nn++){
        int xmin = int(transfor_gt_bboxes[nn].xmin * Crop_Width);
        int xmax = int(transfor_gt_bboxes[nn].xmax * Crop_Width);
        int ymin = int(transfor_gt_bboxes[nn].ymin * Crop_Height);
        int ymax = int(transfor_gt_bboxes[nn].ymax * Crop_Height);
        cv::rectangle(cropImage, cv::Point2i(xmin, ymin), cv::Point2i(xmax, ymax), cv::Scalar(255,0,0), 1, 1, 0);
      }
      cv::imwrite(saved_img_name, cropImage);
    }
  }

	return 1;
}
