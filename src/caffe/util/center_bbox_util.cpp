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

#include "caffe/util/center_bbox_util.hpp"

namespace caffe {
template<typename Dtype>
Dtype gaussian_radius(const Dtype heatmap_width, const Dtype heatmap_height, const Dtype min_overlap){

  Dtype a1  = Dtype(1.0);
  Dtype b1  = (heatmap_width + heatmap_height);
  Dtype c1  = heatmap_width * heatmap_height * (1 - min_overlap) / (1 + min_overlap);
  Dtype sq1 = std::sqrt(b1 * b1 - 4 * a1 * c1);
  Dtype r1  = (b1 + sq1) / 2;

  Dtype a2  = Dtype(4.0);
  Dtype b2  = 2 * (heatmap_height + heatmap_width);
  Dtype c2  = (1 - min_overlap) * heatmap_width * heatmap_height;
  Dtype sq2 = std::sqrt(b2 * b2 - 4 * a2 * c2);
  Dtype r2  = (b2 + sq2) / 2;

  Dtype a3  = Dtype(4 * min_overlap);
  Dtype b3  = -2 * min_overlap * (heatmap_height + heatmap_width);
  Dtype c3  = (min_overlap - 1) * heatmap_width * heatmap_height;
  Dtype sq3 = std::sqrt(b3 * b3 - 4 * a3 * c3);
  Dtype r3  = (b3 + sq3) / 2;
  return std::min(std::min(r1, r2), r3);
}

template float gaussian_radius(const float heatmap_width, const float heatmap_height, const float min_overlap);
template double gaussian_radius(const double heatmap_width, const double heatmap_height, const double min_overlap);


#ifdef USE_OPENCV

cv::Mat gaussian2D(const int height, const int width, const float sigma){
  int half_width = (width - 1) / 2;
  int half_height = (height - 1) /2 ; 
  cv::Mat heatmap(cv::Size(width, height), CV_32FC1, cv::Scalar(0));
  for(int i = 0; i < heatmap.rows; i++){
    float *data = heatmap.ptr<float>(i);
    int x = i - half_width;
    for(int j = 0; j < heatmap.cols; j++){
      int y = j - half_height;
      data[j] = std::exp(-(x*x + y*y)/2);
      if(data[j] < 0)
        data[j] = 0;
    }
  }
  return heatmap;
}

void draw_umich_gaussian(cv::Mat heatmap, float center_x, float center_y, float radius, int k = 1){
  float diameter = 2 * radius + 1;
  cv::Mat gaussian = gaussian2D(int(diameter), int(diameter), float(diameter / 6));
  int height = heatmap.rows, width = heatmap.cols;
  int left = std::min(int(center_x), int(radius)), right = std::min(int(width - center_x), int(radius) + 1);
  int top = std::min(int(center_y), int(radius)), bottom = std::min(int(height - center_y), int(radius) + 1);
  if(left + right > 0 && top + bottom > 0){
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
}



#endif  // USE_OPENCV

}  // namespace caffe
