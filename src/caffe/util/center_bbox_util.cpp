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

namespace caffe {
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

template <typename Dtype>
void EncodeCenteGroundTruthAndPredictions(Dtype* gt_loc_data, Dtype* pred_loc_data,
                                const int output_width, const int output_height, 
                                bool share_location, const Dtype* channel_loc_data,
                                const int num_channels, std::map<int, vector<NormalizedBBox> > all_gt_bboxes){
  std::map<int, vector<NormalizedBBox> > ::iterator iter;
  int count = 0;
  CHECK_EQ(num_channels, 4);
  for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
    int batch_id = iter->first;
    vector<NormalizedBBox> gt_bboxes = iter->second;
    for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
      const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
      const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
      const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
      const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
      Dtype center_x = Dtype((xmin + xmax) / 2);
      Dtype center_y = Dtype((ymin + ymax) / 2);
      int inter_center_x = static_cast<int> (center_x);
      int inter_center_y = static_cast<int> (center_y);
      Dtype diff_x = center_x - inter_center_x;
      Dtype diff_y = center_y - inter_center_y;
      //Dtype width = gt_bboxes[ii].xmax() - gt_bboxes[ii].xmin();
      //Dtype height = gt_bboxes[ii].ymax() - gt_bboxes[ii].ymin();
      Dtype width = xmax - xmin;
      Dtype height = ymax - ymin;
      
      int dimScale = output_height * output_width;
      int x_loc_index = batch_id * num_channels * dimScale
                                + 0 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      int y_loc_index = batch_id * num_channels * dimScale 
                                + 1 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      int width_loc_index = batch_id * num_channels * dimScale
                                + 2 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      int height_loc_index = batch_id * num_channels * dimScale 
                                + 3 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      gt_loc_data[count * num_channels + 0] = diff_x;
      gt_loc_data[count * num_channels + 1] = diff_y;
      gt_loc_data[count * num_channels + 2] = std::log(width);
      gt_loc_data[count * num_channels + 3] = std::log(height);
      pred_loc_data[count * num_channels + 0] = channel_loc_data[x_loc_index];
      pred_loc_data[count * num_channels + 1] = channel_loc_data[y_loc_index];
      pred_loc_data[count * num_channels + 2] = channel_loc_data[width_loc_index];
      pred_loc_data[count * num_channels + 3] = channel_loc_data[height_loc_index];
      ++count;
      #if 0
      LOG(INFO)<<"diff_x: "<<diff_x <<", diff_y: "<<diff_y <<", bbox width : "<<std::log(width)<<", bbox height: "<<std::log(height);
      LOG(INFO)<<"##pred diff_x: "<<loc_data[x_loc_index]<<", pred diff_y: "<<loc_data[y_loc_index]
                                  <<", wh_data_w: "<<wh_data[width_loc_index]<<", wh_data_h: "<<wh_data[height_loc_index];
      LOG(INFO)<<"...";
      #endif
    }
  }
}
template void EncodeCenteGroundTruthAndPredictions(float* gt_loc_data, float* pred_loc_data,
                                const int output_width, const int output_height, 
                                bool share_location, const float* channel_loc_data,
                                const int num_channels, std::map<int, vector<NormalizedBBox> > all_gt_bboxes);
template void EncodeCenteGroundTruthAndPredictions(double* gt_loc_data, double* pred_loc_data,
                                const int output_width, const int output_height, 
                                bool share_location, const double* channel_loc_data,
                                const int num_channels, std::map<int, vector<NormalizedBBox> > all_gt_bboxes);                              

template <typename Dtype>
void CopyDiffToBottom(const Dtype* pre_diff, const int output_width, 
                                const int output_height, 
                                bool share_location, Dtype* bottom_diff, const int num_channels,
                                std::map<int, vector<NormalizedBBox> > all_gt_bboxes){
  std::map<int, vector<NormalizedBBox> > ::iterator iter;
  int count = 0;
  CHECK_EQ(num_channels, 4);
  for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
    int batch_id = iter->first;
    vector<NormalizedBBox> gt_bboxes = iter->second;
    for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
      const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
      const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
      const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
      const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
      Dtype center_x = Dtype((xmin + xmax) / 2);
      Dtype center_y = Dtype((ymin + ymax) / 2);
      int inter_center_x = static_cast<int> (center_x);
      int inter_center_y = static_cast<int> (center_y);
      int dimScale = output_height * output_width;
      int x_loc_index = batch_id * num_channels * dimScale
                                + 0 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      int y_loc_index = batch_id * num_channels * dimScale 
                                + 1 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      int width_loc_index = batch_id * num_channels * dimScale
                                + 2 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      int height_loc_index = batch_id * num_channels * dimScale 
                                + 3 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      bottom_diff[x_loc_index] = pre_diff[count * num_channels + 0];
      bottom_diff[y_loc_index] = pre_diff[count * num_channels + 1];
      bottom_diff[width_loc_index] = pre_diff[count * num_channels + 2];
      bottom_diff[height_loc_index] = pre_diff[count * num_channels + 3];
      ++count;
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

void nms(std::vector<CenterNetInfo>& input, std::vector<CenterNetInfo>* output, float nmsthreshold,int type)
{
	if (input.empty()) {
		return;
	}
	std::sort(input.begin(), input.end(),
		[](const CenterNetInfo& a, const CenterNetInfo& b)
		{
			return a.score() > b.score();
		});

	float IOU = 0.f;
	float maxX = 0.f;
	float maxY = 0.f;
	float minX = 0.f;
	float minY = 0.f;
	std::vector<int> vPick;
	std::vector<pair<float, int> > vScores;
	const int num_boxes = input.size();
	for (int i = 0; i < num_boxes; ++i) {
		vScores.push_back(std::pair<float, int>(input[i].score(), i));
	}

  while (vScores.size() != 0) {
    const int idx = vScores.front().second;
    bool keep = true;
    for (int k = 0; k < vPick.size(); ++k) {
      if (keep) {
        const int kept_idx = vPick[k];
        maxX = std::max(input[idx].xmin(), input[kept_idx].xmin());
        maxY = std::max(input[idx].ymin(), input[kept_idx].ymin());
        minX = std::min(input[idx].xmax(), input[kept_idx].xmax());
        minY = std::min(input[idx].ymax(), input[kept_idx].ymax());
        //maxX1 and maxY1 reuse 
        maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
        maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
        //IOU reuse for the area of two bbox
        IOU = maxX * maxY;
        if (type==NMS_UNION)
          IOU = IOU / (input[idx].area() + input[kept_idx].area() - IOU);
        else if (type == NMS_MIN) {
          IOU = IOU / ((input[idx].area() < input[kept_idx].area()) ? input[idx].area() : input[kept_idx].area());
        }
        keep = IOU <= nmsthreshold;
      } else {
        break;
      }
    }
    if (keep) {
      vPick.push_back(idx);
    }
    vScores.erase(vScores.begin());
  }
	for (unsigned i = 0; i < vPick.size(); i++) {
		output->push_back(input[vPick[i]]);
	}
}

template <typename Dtype>
void get_topK(const Dtype* keep_max_data, const Dtype* loc_data, const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > >* results
                  , const int loc_channels, Dtype conf_thresh, Dtype nms_thresh){
  std::vector<CenterNetInfo > batch_result;
  int dim = classes * output_width * output_height;
  CHECK_EQ(loc_channels, 4);
  for(int i = 0; i < num_batch; i++){
    std::vector<CenterNetInfo > batch_temp;
    batch_result.clear();
    for(int c = 0 ; c < classes; c++){
      int dimScale = output_width * output_height;
      for(int h = 0; h < output_height; h++){
        for(int w = 0; w < output_width; w++){
          int index = i * dim + c * dimScale + h * output_width + w;
          if(keep_max_data[index] > conf_thresh && keep_max_data[index] < 1){
            int x_loc_index = i * loc_channels * dimScale + h * output_width + w;
            int y_loc_index = i * loc_channels * dimScale + dimScale + h * output_width + w;
            int width_loc_index = i * loc_channels * dimScale + 2 * dimScale + h * output_width + w;
            int height_loc_index = i * loc_channels * dimScale + 3 * dimScale + h * output_width + w;
            Dtype center_x = (w + loc_data[x_loc_index]) * 4;
            Dtype center_y = (h + loc_data[y_loc_index]) * 4;
            Dtype width = std::exp(loc_data[width_loc_index]) * 4 ;
            Dtype height = std::exp(loc_data[height_loc_index]) * 4 ;
            Dtype xmin = std::min(std::max(center_x - Dtype(width / 2), Dtype(0.f)), Dtype(4 * output_width));
            Dtype xmax = std::min(std::max(center_x + Dtype(width / 2), Dtype(0.f)), Dtype(4 * output_width));
            Dtype ymin = std::min(std::max(center_y - Dtype(height / 2), Dtype(0.f)), Dtype(4 * output_height));
            Dtype ymax = std::min(std::max(center_y + Dtype(height / 2), Dtype(0.f)), Dtype(4 * output_height));
            CenterNetInfo temp_result;
            temp_result.set_class_id(c);
            temp_result.set_score(keep_max_data[index]);
            temp_result.set_xmin(xmin);
            temp_result.set_xmax(xmax);
            temp_result.set_ymin(ymin);
            temp_result.set_ymax(ymax);
            temp_result.set_area(width * height);
            batch_temp.push_back(temp_result);
          } 
        }
      }
    }
    nms(batch_temp, &batch_result, nms_thresh);
    for(unsigned j = 0 ; j < batch_result.size(); j++){
      batch_result[j].set_xmin(batch_result[j].xmin() / (4 * output_width));
      batch_result[j].set_xmax(batch_result[j].xmax() / (4 * output_width));
      batch_result[j].set_ymin(batch_result[j].ymin() / (4 * output_height));
      batch_result[j].set_ymax(batch_result[j].ymax() / (4 * output_height));
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
                  , std::map<int, std::vector<CenterNetInfo > >* results
                  , const int loc_channels, float conf_thresh, float nms_thresh);
template void get_topK(const double* keep_max_data, const double* loc_data, const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > >* results
                  , const int loc_channels,  double conf_thresh, double nms_thresh);


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
        //if(heatmap[heatmap_index] == Dtype(1.00))
        //  LOG(INFO)<<"heatmap: "<< heatmap[heatmap_index];
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
void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, Dtype* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height){
  std::map<int, vector<NormalizedBBox> > ::iterator iter;
  count_gt = 0;
  
  for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
    int batch_id = iter->first;
    vector<NormalizedBBox> gt_bboxes = iter->second;
    for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
      std::vector<Dtype> heatmap((output_width *output_height), Dtype(0.));
      const int class_id = gt_bboxes[ii].label();
      Dtype *classid_heap = gt_heatmap + (batch_id * num_classes_ + (class_id - 1)) * output_width * output_height;
      const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
      const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
      const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
      const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
      const Dtype width = Dtype(xmax - xmin);
      const Dtype height = Dtype(ymax - ymin);
      Dtype radius = gaussian_radius(height, width, Dtype(0.3));
      radius = std::max(0, int(radius));
      int center_x = static_cast<int>(Dtype((xmin + xmax) / 2));
      int center_y = static_cast<int>(Dtype((ymin + ymax) / 2));
      #if 0
      LOG(INFO)<<"batch_id: "<<batch_id<<", class_id: "
                <<class_id<<", radius: "<<radius<<", center_x: "
                <<center_x<<", center_y: "<<center_y<<", output_height: "
                <<output_height<<", output_width: "<<output_width
                <<", bbox_width: "<<width<<", bbox_height: "<<height;
      #endif
      draw_umich_gaussian( heatmap, center_x, center_y, radius, output_height, output_width );
      transferCVMatToBlobData(heatmap, classid_heap);
      count_gt++;
    }
  }
  #if 0
  count_one = 0;
  int count_no_one = 0;
  for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
    int batch_id = iter->first;
    vector<NormalizedBBox> gt_bboxes = iter->second;
    for(int c = 0; c < num_classes_; c++){
      for(int h = 0 ; h < output_height; h++){
        for(int w = 0; w < output_width; w++){
          int index = batch_id * num_classes_ * output_height * output_width + c * output_height * output_width + h * output_width + w;  
          if(gt_heatmap[index] == 1.f){
            count_one++;
          }else{
            count_no_one++;
          }
        }
      }
    }
  }
  LOG(INFO)<<"count_no_one: "<<count_no_one<<", count_one: "<<count_one<<", count_gt: "<<count_gt;
  #endif
}
template void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, float* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);
template void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, double* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);



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
Dtype YOloSigmoid(Dtype x){
	return 1. / (1. + exp(-x));
}

template double YOloSigmoid(double x);
template float YOloSigmoid(float x);

// 置信度得分，用逻辑回归来做，loss_delta梯度值，既前向又后向
template <typename Dtype>
void EncodeYoloObject(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int net_width, const int net_height,
                          Dtype* channel_pred_data,
                          std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                          std::vector<int> mask_bias, std::vector<std::pair<int, int> >bias_scale, 
                          Dtype* bottom_diff, Dtype ignore_thresh){
  CHECK_EQ(net_height, net_width);
  int stride_channel = 5 + num_classes;
  int stride_feature = net_height / output_height;
  int dimScale = output_height * output_width;
  float avg_iou = 0;
  float recall = 0;
  float recall75 = 0;
  float avg_cat = 0;
  float avg_obj = 0;
  float avg_anyobj = 0;
  int count = 0;
  int class_count = 0;
  CHECK_EQ(num_channels, (5 + num_classes) * mask_bias.size()) << "num_channels shoule be set to including bias_x, bias_y, width, height, object_confidence and classes";
  for(int b = 0; b < batch_size; b++){
    for(unsigned m = 0; m < mask_bias.size(); m++){
      int x_index = b * num_channels * dimScale
                                  + (m * stride_channel + 0)* dimScale;
      for(int i = 0; i < 2 * dimScale; i++)
        channel_pred_data[x_index + i] = YOloSigmoid(channel_pred_data[x_index + i]);
      int object_index = b * num_channels * dimScale
                                  + (m * stride_channel + 4)* dimScale;
      for(int i = 0; i < (num_classes + 1) * dimScale; i++){
        channel_pred_data[object_index + i] = YOloSigmoid(channel_pred_data[object_index + i]);
      }
    }
  }
  for(int b = 0; b < batch_size; b++){
    vector<NormalizedBBox> gt_bboxes = all_gt_bboxes.find(b)->second;
    for(int h = 0; h < output_height; h++){
      for(int w = 0; w < output_width; w++){
        for(unsigned m = 0; m < mask_bias.size(); m++){
          //LOG(INFO)<<"output_dim: "<<output_height<<", mask: "<<mask_bias[m];
          int x_index = b * num_channels * dimScale
                                    + (m * stride_channel + 0)* dimScale + h * output_width + w;
          int y_index = b * num_channels * dimScale 
                                    + (m * stride_channel + 1)* dimScale + h * output_width + w;
          int width_index = b * num_channels * dimScale
                                    + (m * stride_channel + 2)* dimScale + h * output_width + w;
          int height_index = b * num_channels * dimScale 
                                    + (m * stride_channel + 3)* dimScale + h * output_width + w;
          int object_index = b * num_channels * dimScale 
                                    + (m * stride_channel + 4)* dimScale + h * output_width + w;
          NormalizedBBox predBox;
          float bb_xmin = (channel_pred_data[x_index] + w) / output_width;
          float bb_ymin = (channel_pred_data[y_index] + h) / output_height;
          float bb_width = (float)std::exp(channel_pred_data[width_index]) 
                                                          * bias_scale[mask_bias[m]].first / net_width;
          float bb_height = (float)std::exp(channel_pred_data[height_index]) 
                                                          * bias_scale[mask_bias[m]].second / net_height;
          predBox.set_xmin(bb_ymin);
          predBox.set_xmax(bb_xmin + bb_width);
          predBox.set_ymin(bb_ymin);
          predBox.set_ymax(bb_ymin + bb_height);
          float best_iou = 0;
          int best_t = 0;
          for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
            float iou = YoloBBoxIou(predBox, gt_bboxes[ii]);
            if (iou > best_iou) {
              best_iou = iou;
              best_t = ii;
            }
          }
          avg_anyobj += channel_pred_data[object_index];
          bottom_diff[object_index] = 0 - channel_pred_data[object_index];
          if(best_iou > ignore_thresh){
            bottom_diff[object_index] = 0;
          }
        }
      }
    }
    for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
      const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
      const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
      const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
      const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
      float best_iou = 0.f;
      int best_mask_scale = 0;
      for(unsigned m = 0; m < bias_scale.size(); m++){
        NormalizedBBox anchor_bbox ; 
        NormalizedBBox shfit_gt_bbox;
        Dtype shift_x_min = 0 - Dtype((xmax - xmin) / 2 / output_width);
        Dtype shift_x_max = 0 + Dtype((xmax - xmin) / 2 / output_width);
        Dtype shift_y_min = 0 - Dtype((ymax - ymin) / 2 / output_height);
        Dtype shift_y_max = 0 + Dtype((ymax - ymin) / 2 / output_height);
        shfit_gt_bbox.set_xmin(shift_x_min);
        shfit_gt_bbox.set_xmax(shift_x_max);
        shfit_gt_bbox.set_ymin(shift_y_min);
        shfit_gt_bbox.set_ymax(shift_y_max);
        float bias_width = bias_scale[m].first;
        float bias_height = bias_scale[m].second;
        float bias_xmin = 0 - (float)bias_width / 2 / net_width;
        float bias_ymin = 0 - (float)bias_height / 2 / net_height;
        float bias_xmax = 0 + (float)bias_width / 2 / net_width;
        float bias_ymax = 0 + (float)bias_height / 2 / net_height;
        anchor_bbox.set_xmin(bias_xmin);
        anchor_bbox.set_xmax(bias_xmax);
        anchor_bbox.set_ymin(bias_ymin);
        anchor_bbox.set_ymax(bias_ymax);
        float iou = YoloBBoxIou(shfit_gt_bbox, anchor_bbox);
        if (iou > best_iou){
          best_iou = iou;
          best_mask_scale = m;
        }
      }
      int mask_n = int_index(mask_bias, best_mask_scale, mask_bias.size());
      if(mask_n > 0){
        Dtype center_x = Dtype((xmin + xmax) / 2);
        Dtype center_y = Dtype((ymin + ymax) / 2);
        int inter_center_x = static_cast<int> (center_x);
        int inter_center_y = static_cast<int> (center_y);
        Dtype diff_x = center_x - inter_center_x;
        Dtype diff_y = center_y - inter_center_y;
        Dtype width = std::log((xmax - xmin) * stride_feature / bias_scale[best_mask_scale].first);
        Dtype height = std::log((ymax - ymin) * stride_feature / bias_scale[best_mask_scale].second);
        
        int x_index = b * num_channels * dimScale + (mask_n * stride_channel + 0)* dimScale
                                + inter_center_y * output_width + inter_center_x;
        int y_index = b * num_channels * dimScale + (mask_n * stride_channel + 1)* dimScale
                                  + inter_center_y * output_width + inter_center_x;
        int width_index = b * num_channels * dimScale + (mask_n * stride_channel + 2)* dimScale
                                  + inter_center_y * output_width + inter_center_x;
        int height_index = b * num_channels * dimScale + (mask_n * stride_channel + 3)* dimScale
                                  + inter_center_y * output_width + inter_center_x;
        int object_index = b * num_channels * dimScale + (mask_n * stride_channel + 4)* dimScale
                                  + inter_center_y * output_width + inter_center_x;
        float delta_scale = 2 - (float)(xmax - xmin) * (ymax - ymin) / (output_height * output_width);
        bottom_diff[x_index] = delta_scale * (diff_x - channel_pred_data[x_index]);
        bottom_diff[y_index] = delta_scale * (diff_y - channel_pred_data[y_index]);
        bottom_diff[width_index] = delta_scale * (width - channel_pred_data[width_index]);
        bottom_diff[height_index] = delta_scale * (height - channel_pred_data[height_index]);
        bottom_diff[object_index] = 1 - channel_pred_data[object_index];
        avg_obj +=  channel_pred_data[object_index];
        // class score
        // 特殊情况，face数据集，包含了背景目标，而实际上不需要背景目标，所以减一
        int class_lable = gt_bboxes[ii].label() - 1; 
        int class_index = b * num_channels * dimScale + (mask_n * stride_channel + 5)* dimScale
                                  + inter_center_y * output_width + inter_center_x;
        if ( bottom_diff[class_index]){
          bottom_diff[class_index + class_lable * dimScale] = 1 - channel_pred_data[class_index + class_lable * dimScale];
          avg_cat += channel_pred_data[class_index + class_lable * dimScale];
        }else{
          for(int c = 0; c < num_classes; c++){
            bottom_diff[class_index + c * dimScale] = ((c == class_lable)?1 : 0) - channel_pred_data[class_index + c * dimScale];
            if(c == class_lable) 
              avg_cat += channel_pred_data[class_index + c * dimScale];
          }
        }

        ++count;
        ++class_count;
        NormalizedBBox predBox;
        float bb_xmin = (channel_pred_data[x_index] + inter_center_x) / output_width;
        float bb_ymin = (channel_pred_data[y_index] + inter_center_y) / output_height;
        float bb_width = (float)std::exp(channel_pred_data[width_index]) 
                                                        * bias_scale[best_mask_scale].first / net_width;
        float bb_height = (float)std::exp(channel_pred_data[height_index]) 
                                                        * bias_scale[best_mask_scale].second / net_height;
        predBox.set_xmin(bb_ymin);
        predBox.set_xmax(bb_xmin + bb_width);
        predBox.set_ymin(bb_ymin);
        predBox.set_ymax(bb_ymin + bb_height);
        float iou = YoloBBoxIou(predBox, gt_bboxes[ii]);
        if(iou > .5) recall += 1;
        if(iou > .75) recall75 += 1;
        avg_iou += iou;
        
      }
    } 
  }
  LOG(INFO)<<"Region "<<output_width<<" Avg IOU: "<<avg_iou/count<<", Class: "<<avg_cat/class_count
                      <<", Obj: "<<avg_obj/count<<", No obj: "<<avg_anyobj/(dimScale*mask_bias.size()*batch_size)
                      <<", .5R: "<<recall/count<<", .75R: "<<recall75/count<<", count: "<<count;
}
template void EncodeYoloObject(const int batch_size, const int num_channels, const int num_classes,
                              const int output_width, const int output_height, 
                              const int net_width, const int net_height,
                              float* channel_pred_data,
                              std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                              std::vector<int> mask_bias, std::vector<std::pair<int, int> >bias_scale, 
                              float* bottom_diff, float ignore_thresh);
template void EncodeYoloObject(const int batch_size, const int num_channels, const int num_classes,
                              const int output_width, const int output_height, 
                              const int net_width, const int net_height,
                              double* channel_pred_data,
                              std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                              std::vector<int> mask_bias, std::vector<std::pair<int, int> >bias_scale, 
                              double* bottom_diff, double ignore_thresh);



template <typename Dtype>
void GetYoloGroundTruth(const Dtype* gt_data, int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      std::map<int, vector<NormalizedBBox> >* all_gt_bboxes, int batch_size){
  all_gt_bboxes->clear();
  for(int b = 0; b < batch_size; b++){
    for (int i = 0; i < num_gt; ++i) {
      int start_idx = b * num_gt * 8 +  i * 8;
      int item_id = gt_data[start_idx];
      if (item_id == -1) {
        continue;
      }
      int label = gt_data[start_idx + 1];
      CHECK_NE(background_label_id, label)
          << "Found background label in the dataset.";
      bool difficult = static_cast<bool>(gt_data[start_idx + 7]);
      if (!use_difficult_gt && difficult) {
        continue;
      }
      NormalizedBBox bbox;
      bbox.set_label(label);
      bbox.set_xmin(gt_data[start_idx + 3]);
      bbox.set_ymin(gt_data[start_idx + 4]);
      bbox.set_xmax(gt_data[start_idx + 5]);
      bbox.set_ymax(gt_data[start_idx + 6]);
      bbox.set_difficult(difficult);
      float bbox_size = BBoxSize(bbox);
      bbox.set_size(bbox_size);
      (*all_gt_bboxes)[item_id].push_back(bbox);
    }
  }
}
template void GetYoloGroundTruth(const float* gt_data, int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      std::map<int, vector<NormalizedBBox> >* all_gt_bboxes, int batch_size);

template void GetYoloGroundTruth(const double* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      std::map<int, vector<NormalizedBBox> >* all_gt_bboxes, int batch_size);
}  // namespace caffe
