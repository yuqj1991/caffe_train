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
  //LOG(INFO)<<r1<<", "<<r2<<", "<<r3;
  return std::min(std::min(r1, r2), r3);
}

template float gaussian_radius(const float heatmap_width, const float heatmap_height, const float min_overlap);
template double gaussian_radius(const double heatmap_width, const double heatmap_height, const double min_overlap);

template <typename Dtype>
void EncodeCenteGroundTruthAndPredictions(const Dtype* loc_data, const Dtype* wh_data, 
                                const int output_width, const int output_height, 
                                bool share_location, Dtype* pred_loc_data, Dtype* pred_wh_data, 
                                const int num_channels, Dtype* gt_loc_data, Dtype* gt_wh_data,
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
      Dtype center_x = Dtype((xmin + xmax) / 2);
      Dtype center_y = Dtype((ymin + ymax) / 2);
      int inter_center_x = static_cast<int> (center_x);
      int inter_center_y = static_cast<int> (center_y);
      Dtype diff_x = center_x - inter_center_x;
      Dtype diff_y = center_y - inter_center_y;
      Dtype width = xmax - xmin;
      Dtype height = ymax - ymin;
      #if 0
      if(count == 25)
      LOG(INFO)<<"center_x: "<<center_x<<", center_y: "<<center_y
               <<", inter_center_x: "<<inter_center_x<<", inter_center_y: "<<inter_center_y
               <<", diff_x: "<<diff_x<<", diff_y: "<<diff_y
               <<", width: "<<width<<", height: "<<height;
      #endif
      int dimScale = output_height * output_width;
      int x_loc_index = batch_id * num_channels * dimScale
                                + 0 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      int y_loc_index = batch_id * num_channels * dimScale 
                                + 1 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      int width_loc_index = batch_id * num_channels * dimScale
                                + 0 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      int height_loc_index = batch_id * num_channels * dimScale 
                                + 1 * dimScale
                                + inter_center_y * output_width + inter_center_x;
      gt_loc_data[count * num_channels + 0] = diff_x;
      gt_loc_data[count * num_channels + 1] = diff_y;
      gt_wh_data[count * num_channels + 0] = log(width);
      gt_wh_data[count * num_channels + 1] = log(height);
      pred_loc_data[count * num_channels + 0] = loc_data[x_loc_index];
      pred_loc_data[count * num_channels + 1] = loc_data[y_loc_index];
      pred_wh_data[count * num_channels + 0] = wh_data[width_loc_index];
      pred_wh_data[count * num_channels + 1] = wh_data[height_loc_index];
      count++;
      #if 0
      LOG(INFO)<<"diff_x: "<<diff_x<<", diff_y: "<<diff_y<<", width: "<<width  <<", height: "<<height;
      #endif
    }
  }
}
template void EncodeCenteGroundTruthAndPredictions(const float* loc_data, const float* wh_data, 
                                const int output_width, const int output_height, 
                                bool share_location, float* pred_loc_data, float* pred_wh_data, 
                                const int num_channels, float* gt_loc_data, float* gt_wh_data,
                                std::map<int, vector<NormalizedBBox> > all_gt_bboxes);
template void EncodeCenteGroundTruthAndPredictions(const double* loc_data, const double* wh_data, 
                                const int output_width, const int output_height, 
                                bool share_location, double* pred_loc_data, double* pred_wh_data, 
                                const int num_channels, double* gt_loc_data, double* gt_wh_data,
                                std::map<int, vector<NormalizedBBox> > all_gt_bboxes);                              

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
      bottom_diff[x_loc_index] = pre_diff[count * num_channels + 0];
      bottom_diff[y_loc_index] = pre_diff[count * num_channels + 1];
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

void nms(std::vector<CenterNetInfo>& input, std::vector<CenterNetInfo>& output, float nmsthreshold,int type)
{
	if (input.empty()) {
		return;
	}
	std::sort(input.begin(), input.end(),
		[](const CenterNetInfo& a, const CenterNetInfo& b)
		{
			return a.score < b.score;
		});

	float IOU = 0.f;
	float maxX = 0.f;
	float maxY = 0.f;
	float minX = 0.f;
	float minY = 0.f;
	std::vector<int> vPick;
	int nPick = 0;
	std::multimap<float, int> vScores;
	const int num_boxes = input.size();
	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i) {
		vScores.insert(std::pair<float, int>(input[i].score, i));
	}
	while (vScores.size() > 0) {
		int last = vScores.rbegin()->second;
		vPick[nPick] = last;
		nPick += 1;
		for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
			int it_idx = it->second;
			maxX = std::max(input.at(it_idx).xmin, input.at(last).xmin);
			maxY = std::max(input.at(it_idx).ymin, input.at(last).ymin);
			minX = std::min(input.at(it_idx).xmax, input.at(last).xmax);
			minY = std::min(input.at(it_idx).ymax, input.at(last).ymax);
			//maxX1 and maxY1 reuse 
			maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
			//IOU reuse for the area of two bbox
			IOU = maxX * maxY;
			if (type==NMS_UNION)
				IOU = IOU / (input.at(it_idx).area + input.at(last).area - IOU);
			else if (type == NMS_MIN) {
				IOU = IOU / ((input.at(it_idx).area < input.at(last).area) ? input.at(it_idx).area : input.at(last).area);
			}
			if (IOU > nmsthreshold) {
				it = vScores.erase(it);
			}
			else {
				it++;
			}
		}
	}

	vPick.resize(nPick);
	output.resize(nPick);
	for (int i = 0; i < nPick; i++) {
		output[i] = input[vPick[i]];
	}
}

template <typename Dtype>
void get_topK(const Dtype* keep_max_data, const Dtype* loc_data, const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > >* results
                  , const int loc_channels, Dtype conf_thresh, Dtype nms_thresh){
  std::vector<CenterNetInfo > batch_result;
  int dim = classes * output_width * output_height;
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
            Dtype width = std::exp(loc_data[width_loc_index]) * 4;
            Dtype height = std::exp(loc_data[height_loc_index]) * 4;
            CenterNetInfo temp_result = {
              .class_id = c,
              .score = keep_max_data[index],
              .xmin = Dtype(center_x - Dtype(width / 2) > 0 ? center_x - Dtype(width / 2) : 0),
              .ymin = Dtype(center_y - Dtype(height / 2) > 0 ? center_y - Dtype(height / 2) :0),
              .xmax = Dtype(center_x + Dtype(width / 2) < 4 * output_width ? center_x + Dtype(width / 2) : 4 * output_width),
              .ymax = Dtype(center_y + Dtype(height / 2) < 4 * output_height ? center_y + Dtype(height / 2) : 4 * output_height),
              .area = Dtype(width * height)
            };
            batch_temp.push_back(temp_result);
          } 
        }
      }
    }
    nms(batch_temp, batch_result, nms_thresh);
    LOG(INFO)<<"get_TopK batch_id "<<i << " detection results: "<<batch_result.size();
    for(unsigned j = 0 ; j < batch_result.size(); j++){
      batch_result[j].xmin = float(batch_result[j].xmin / (4 * output_width));
      batch_result[j].xmax = float(batch_result[j].xmax / (4 * output_width));
      batch_result[j].ymin = float(batch_result[j].ymin / (4 * output_height));
      batch_result[j].ymax = float(batch_result[j].ymax / (4 * output_height));
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

#ifdef USE_OPENCV

cv::Mat gaussian2D(const int height, const int width, const float sigma){
  int half_width = (width - 1) / 2;
  int half_height = (height - 1) / 2;
  cv::Mat heatmap(cv::Size(width, height), CV_32FC1, cv::Scalar(0));
  CHECK_EQ(height, heatmap.rows);
  CHECK_EQ(width, heatmap.cols);
  for(int i = 0; i < height; i++){
    float *data = heatmap.ptr<float>(i);
    int x = i - half_height;
    for(int j = 0; j < width; j++){
      int y = j - half_width;
      data[j] = std::exp(float(-(x*x + y*y) / (2* sigma * sigma)));
      if(data[j] < 0.00000000005)
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
      Dtype radius = gaussian_radius(width, height, Dtype(0.3));
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
      draw_umich_gaussian( heatmap, center_x, center_y, radius );
      transferCVMatToBlobData(heatmap, classid_heap);
      count_gt++;
    }
  }
  #if 0
  for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
    int batch_id = iter->first;
    vector<NormalizedBBox> gt_bboxes = iter->second;
    for(int c = 0; c < num_classes_; c++){
      for(int h = 0 ; h < output_height; h++){
        for(int w = 0; w < output_width; w++){
          int index = batch_id * num_classes_ * output_height * output_width + c * output_height * output_width + h * output_width + w;  
          if(gt_heatmap[index] == 1.f){
            count_one++;
            //LOG(INFO)<<"heatmap center_x: "<< w << ", heatmap center_y; "<< h << ", value: "<<gt_heatmap[index];
          }
        }
      }
    }
  }
  //LOG(INFO)<<"count_no_zero: "<<count_no_zero<<", count_zero: "<<count_zero;
  if(std::abs(count_one - count_gt) > 1)
    LOG(FATAL) << "count_one is not equal to count_gt: " << count_one <<", "<<count_gt;
  #endif
}
template void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, float* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);
template void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, double* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);
#endif  // USE_OPENCV

}  // namespace caffe
