#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


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


using namespace cv;
using namespace std;
struct NormalizedBBox{
   float xmin;
   float ymin;
   float xmax;
   float ymax;
};

int count_no_zero=0;
int count_one = 0;
int count_zero = 0;

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

cv::Mat gaussian2D(const int height, const int width, const float sigma){
  int half_width = (width - 1) / 2;
  int half_height = (height - 1) / 2;
  cv::Mat heatmap(cv::Size(width, height), CV_32FC1, cv::Scalar(0));
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
  for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
    int batch_id = iter->first;
    vector<NormalizedBBox> gt_bboxes = iter->second;
    for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
      cv::Mat heatmap(cv::Size(output_width, output_height), CV_32FC1, cv::Scalar(0));
      const int class_id = 1;
      Dtype *classid_heap = gt_heatmap + (batch_id * num_classes_ + (class_id - 1)) * output_width * output_height;
      const Dtype xmin = gt_bboxes[ii].xmin * output_width;
      const Dtype ymin = gt_bboxes[ii].ymin * output_height;
      const Dtype xmax = gt_bboxes[ii].xmax * output_width;
      const Dtype ymax = gt_bboxes[ii].ymax * output_height;
      const Dtype width = Dtype(xmax - xmin);
      const Dtype height = Dtype(ymax - ymin);
      Dtype radius = gaussian_radius(width, height, Dtype(0.7));
      radius = std::max(0, int(radius));
      int center_x = static_cast<int>(Dtype((xmin + xmax) / 2));
      int center_y = static_cast<int>(Dtype((ymin + ymax) / 2));
      #if 1
      std::cout<<"batch_id: "<<batch_id<<", class_id: "
                <<class_id<<", radius: "<<radius<<", center_x: "
                <<center_x<<", center_y: "<<center_y<<", output_height: "
                <<output_height<<", output_width: "<<output_width
                <<", bbox_width: "<<width<<", bbox_height: "<<height<<std::endl;
      #endif
      draw_umich_gaussian( heatmap, center_x, center_y, radius );
      transferCVMatToBlobData(heatmap, classid_heap);
    }
  }
  #if 1
  for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
    int batch_id = iter->first;
    vector<NormalizedBBox> gt_bboxes = iter->second;
    for(int c = 0; c < num_classes_; c++){
      for(int h = 0 ; h < output_height; h++){
        for(int w = 0; w < output_width; w++){
          int index = batch_id * num_classes_ * output_height * output_width + c * output_height * output_width + h * output_width + w;  
          if(gt_heatmap[index] == 1.f){
            count_one++;
            std::cout<<"heatmap center_x: "<< w << ", heatmap center_y; "<< h << ", value: "<<gt_heatmap[index]<<std::endl;
          }else if(gt_heatmap[index] == 0.f)
          		count_zero++;
          	else if(gt_heatmap[index] < 1 && gt_heatmap[index] >0)
          		count_no_zero++;
        }
      }
    }
  }
  std::cout<<"count_no_zero: "<<count_no_zero<<", count_zero: "<<count_zero<<", count_one: "<<count_one<<std::endl;
  #endif
}
template void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, float* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);
template void GenerateBatchHeatmap(std::map<int, vector<NormalizedBBox> > all_gt_bboxes, double* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);
                              
                              
int main(){
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
