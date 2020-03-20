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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
using namespace cv;
using namespace std;

struct NormalizedBBox{
   float xmin;
   float ymin;
   float xmax;
   float ymax;
};
void GenerateDataAnchorSample(cv::Mat img, 
                                std::vector<int>anchorScale,
                                const vector<NormalizedBBox>& object_bboxes,
                                int resized_height, int resized_width,
                                NormalizedBBox* sampled_bbox, cv::Mat resized_img,
                                const TransformationParameter& trans_param, bool do_resize){
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
    max_resize_val = COMPAREMIN((float)anchorScale[range_idx_size] * 2,
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
    ResizedCropSample(anno_datum, resized_anno_datum, scale, trans_param);
    int Resized_ori_Height = int(scale * img_height);
    int Resized_ori_Width = int(scale * img_width);
    int Resized_bbox_width = int(scale * bbox_width);
    int Resized_bbox_height = int(scale * bbox_height);
    const float Resized_xmin = object_bboxes[object_bbox_index].xmin()*Resized_ori_Width;
    const float Resized_ymin = object_bboxes[object_bbox_index].ymin()*Resized_ori_Height;
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
  }else{
    LOG(FATAL)<<"need to make do resize";
  }
  sampled_bbox->set_xmin(w_off);
  sampled_bbox->set_ymin(h_off);
  sampled_bbox->set_xmax(w_end);
  sampled_bbox->set_ymax(h_end);
}

void GenerateBatchDataAnchorSamples(const AnnotatedDatum& anno_datum,
                                const vector<DataAnchorSampler>& data_anchor_samplers,
                                int resized_height, int resized_width, 
                                NormalizedBBox* sampled_bbox, AnnotatedDatum* resized_anno_datum, 
                                const TransformationParameter& trans_param, bool do_resize) {
  CHECK_EQ(data_anchor_samplers.size(), 1);
  vector<NormalizedBBox> object_bboxes;
  GroupObjectBBoxes(anno_datum, &object_bboxes);
  for (int i = 0; i < data_anchor_samplers.size(); ++i) {
    if (data_anchor_samplers[i].use_original_image()) {
      int found = 0;
      for (int j = 0; j < data_anchor_samplers[i].max_trials(); ++j) {
        if (data_anchor_samplers[i].has_max_sample() &&
            found >= data_anchor_samplers[i].max_sample()) {
          break;
        }
        AnnotatedDatum temp_anno_datum;
        NormalizedBBox temp_sampled_bbox;
        GenerateDataAnchorSample(anno_datum, data_anchor_samplers[i], object_bboxes, resized_height, 
                                resized_width, &temp_sampled_bbox, &temp_anno_datum, trans_param, do_resize);
        if (SatisfySampleConstraint(temp_sampled_bbox, object_bboxes,
                                      data_anchor_samplers[i].sample_constraint())){
          found++;
          resized_anno_datum->CopyFrom(temp_anno_datum);
          sampled_bbox->CopyFrom(temp_sampled_bbox);
        }
      }
      if(found == 0){
        resized_anno_datum->CopyFrom(anno_datum);
        sampled_bbox->set_xmin(0.f);
        sampled_bbox->set_ymin(0.f);
        sampled_bbox->set_xmax(1.f);
        sampled_bbox->set_ymax(1.f);
      }
    }else{
      LOG(FATAL)<<"must use original_image";
    }
  }
  CHECK_GT(resized_anno_datum->datum().channels(), 0)<<"channels: "<<resized_anno_datum->datum().channels();
}

void ResizedCropSample(const cv::Mat& src_img, cv::Mat resized_img, float scale){
  const Datum datum = anno_datum.datum();
  const int img_width = datum.width();
  const int img_height = datum.height();

  // image data
  cv::Mat resized_img;
  int Resized_img_Height = int(img_height * scale);
  int Resized_img_Width = int(img_width * scale);
  cv::resize(cv_img, resized_img, cv::Size(Resized_img_Width, Resized_img_Height), 0, 0,
                cv::INTER_CUBIC);
  EncodeCVMatToDatum(resized_img, "jpg", resized_anno_datum->mutable_datum());
  resized_anno_datum->mutable_datum()->set_label(datum.label());
  resized_anno_datum->set_type(anno_datum.type());
  // labels trans
  if (anno_datum.type() == AnnotatedDatum_AnnotationType_BBOX) {
		// Go through each AnnotationGroup.
    resized_anno_datum->mutable_annotation_group()->CopyFrom(anno_datum.annotation_group());
	} else {
		LOG(FATAL) << "Unknown annotation type.";
	}
  CHECK_GT(resized_anno_datum->datum().channels(), 0);
}
                              
int main(){
	int batch_id = 1;
	NormalizedBBox box_1 = {0.2, 0.2, 0.5, 0.4};
	NormalizedBBox box_2 = {0.6, 0.5, 0.9, 0.9};
	NormalizedBBox box_3 = {0.3, 0.3, 0.8, 0.5};
	std::map<int, vector<NormalizedBBox> > all_gt_bboxes;
	std::vector<NormalizedBBox> box_set;
	box_set.push_back(box_1);
	box_set.push_back(box_2);
	box_set.push_back(box_3);
	all_gt_bboxes.insert(std::make_pair(0, box_set));
	const int output_height = 128;
	const int output_width = 128;
	const int num_classes_ = 1;
	cv::Mat gt_heatmap(cv::Size(output_width, output_height), CV_32FC1, cv::Scalar(0));
	float* gt_heatmap_data =  gt_heatmap.ptr<float>(0);
	GenerateBatchHeatmap(all_gt_bboxes, gt_heatmap_data, num_classes_, output_width, output_height);
	
	/*for(int row = 0; row < output_height; row++){
		float* gt_heatmap_data_ =  gt_heatmap.ptr<float>(row);
		for(int col  = 0; col < output_width; col++){
			printf("%f ", gt_heatmap_data_[col]);
		}
		printf("\n");
	}*/

	cv::imshow("Gaussian", gt_heatmap);
	cv::waitKey(0);
}
