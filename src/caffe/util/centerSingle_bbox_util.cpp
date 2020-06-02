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

#include "caffe/util/centerSingle_bbox_util.hpp"
using namespace std;
namespace caffe {
template <typename Dtype>
void GetLocTruthAndPrediction(Dtype* gt_loc_offest_data, Dtype* pred_loc_offest_data,
                                Dtype* gt_loc_wh_data, Dtype* pred_loc_wh_data,
                                Dtype* gt_lm_data, Dtype* pred_lm_data,
                                const int output_width, const int output_height, 
                                bool share_location, const Dtype* loc_data,
                                const Dtype* wh_data, const Dtype* lm_data,
                                const int loc_channels, const int lm_channels,
                                std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                                bool has_lm){
    std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > ::iterator iter;
    CHECK_EQ(share_location, true);
    int dimScale = output_height * output_width;
    int count = 0;
    int lm_count = 0;
    if(has_lm){
        CHECK_EQ(loc_channels, 2);
        CHECK_EQ(lm_channels, 10);
    }else{
        CHECK_EQ(loc_channels, 2);
        CHECK_EQ(lm_channels, 1);
    }
    for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
        int batch_id = iter->first;
        vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > gt_bboxes = iter->second;
        for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
            const Dtype xmin = gt_bboxes[ii].first.xmin() * output_width;
            const Dtype ymin = gt_bboxes[ii].first.ymin() * output_height;
            const Dtype xmax = gt_bboxes[ii].first.xmax() * output_width;
            const Dtype ymax = gt_bboxes[ii].first.ymax() * output_height;
            Dtype center_x = Dtype((xmin + xmax) / 2);
            Dtype center_y = Dtype((ymin + ymax) / 2);
            int inter_center_x = static_cast<int> (center_x);
            int inter_center_y = static_cast<int> (center_y);
            Dtype diff_x = center_x - inter_center_x;
            Dtype diff_y = center_y - inter_center_y;
            Dtype width = xmax - xmin;
            Dtype height = ymax - ymin;

            int x_loc_index = batch_id * loc_channels * dimScale
                                    + 0 * dimScale + inter_center_y * output_width + inter_center_x;
            int y_loc_index = batch_id * loc_channels * dimScale 
                                    + 1 * dimScale + inter_center_y * output_width + inter_center_x;
            int width_loc_index = batch_id * loc_channels * dimScale
                                    + 0 * dimScale + inter_center_y * output_width + inter_center_x;
            int height_loc_index = batch_id * loc_channels * dimScale 
                                    + 1 * dimScale + inter_center_y * output_width + inter_center_x;
            gt_loc_offest_data[count * 2 + 0] = diff_x;
            gt_loc_offest_data[count * 2 + 1] = diff_y;
            gt_loc_wh_data[count * 2 + 0] = std::log(width);
            gt_loc_wh_data[count * 2 + 1] = std::log(height);
            pred_loc_offest_data[count * 2 + 0] = loc_data[x_loc_index];
            pred_loc_offest_data[count * 2 + 1] = loc_data[y_loc_index];
            pred_loc_wh_data[count * 2 + 0] = wh_data[width_loc_index];
            pred_loc_wh_data[count * 2 + 1] = wh_data[height_loc_index];
            ++count;
            if(has_lm){
                //lm_gt_datas, & lm_pred_datas
                if(gt_bboxes[ii].second.lefteye().x() > 0 && gt_bboxes[ii].second.lefteye().y() > 0 &&
                   gt_bboxes[ii].second.righteye().x() > 0 && gt_bboxes[ii].second.righteye().y() > 0 && 
                   gt_bboxes[ii].second.nose().x() > 0 && gt_bboxes[ii].second.nose().y() > 0 &&
                   gt_bboxes[ii].second.leftmouth().x() > 0 && gt_bboxes[ii].second.leftmouth().y() > 0 &&
                   gt_bboxes[ii].second.rightmouth().x() > 0 && gt_bboxes[ii].second.rightmouth().y() > 0){
                    gt_lm_data[lm_count * 10 + 0] = Dtype((gt_bboxes[ii].second.lefteye().x() * output_width - inter_center_x) / width);
                    gt_lm_data[lm_count * 10 + 1] = Dtype((gt_bboxes[ii].second.lefteye().y() * output_height - inter_center_y) / height);
                    gt_lm_data[lm_count * 10 + 2] = Dtype((gt_bboxes[ii].second.righteye().x() * output_width - inter_center_x) / width);
                    gt_lm_data[lm_count * 10 + 3] = Dtype((gt_bboxes[ii].second.righteye().y() * output_height - inter_center_y) / height);
                    gt_lm_data[lm_count * 10 + 4] = Dtype((gt_bboxes[ii].second.nose().x() * output_width - inter_center_x) / width);
                    gt_lm_data[lm_count * 10 + 5] = Dtype((gt_bboxes[ii].second.nose().y() * output_height - inter_center_y) / height);
                    gt_lm_data[lm_count * 10 + 6] = Dtype((gt_bboxes[ii].second.leftmouth().x() * output_width - inter_center_x) / width);
                    gt_lm_data[lm_count * 10 + 7] = Dtype((gt_bboxes[ii].second.leftmouth().y() * output_height - inter_center_y) / height);
                    gt_lm_data[lm_count * 10 + 8] = Dtype((gt_bboxes[ii].second.rightmouth().x() * output_width - inter_center_x) / width);
                    gt_lm_data[lm_count * 10 + 9] = Dtype((gt_bboxes[ii].second.rightmouth().y() * output_height - inter_center_y) / height);

                    int le_x_index = batch_id * lm_channels * dimScale
                                        + 0 * dimScale + inter_center_y * output_width + inter_center_x;
                    int le_y_index = batch_id * lm_channels * dimScale 
                                        + 1 * dimScale + inter_center_y * output_width + inter_center_x;
                    int re_x_index = batch_id * lm_channels * dimScale
                                        + 2 * dimScale + inter_center_y * output_width + inter_center_x;
                    int re_y_index = batch_id * lm_channels * dimScale 
                                        + 3 * dimScale + inter_center_y * output_width + inter_center_x;
                    int no_x_index = batch_id * lm_channels * dimScale
                                        + 4 * dimScale + inter_center_y * output_width + inter_center_x;
                    int no_y_index = batch_id * lm_channels * dimScale 
                                        + 5 * dimScale + inter_center_y * output_width + inter_center_x;
                    int lm_x_index = batch_id * lm_channels * dimScale
                                        + 6 * dimScale + inter_center_y * output_width + inter_center_x;
                    int lm_y_index = batch_id * lm_channels * dimScale 
                                        + 7 * dimScale + inter_center_y * output_width + inter_center_x;
                    int rm_x_index = batch_id * lm_channels * dimScale
                                        + 8 * dimScale + inter_center_y * output_width + inter_center_x;
                    int rm_y_index = batch_id * lm_channels * dimScale 
                                        + 9 * dimScale + inter_center_y * output_width + inter_center_x;

                    pred_lm_data[lm_count * 10 + 0] = lm_data[le_x_index];
                    pred_lm_data[lm_count * 10 + 1] = lm_data[le_y_index];
                    pred_lm_data[lm_count * 10 + 2] = lm_data[re_x_index];
                    pred_lm_data[lm_count * 10 + 3] = lm_data[re_y_index];
                    pred_lm_data[lm_count * 10 + 4] = lm_data[no_x_index];
                    pred_lm_data[lm_count * 10 + 5] = lm_data[no_y_index];
                    pred_lm_data[lm_count * 10 + 6] = lm_data[lm_x_index];
                    pred_lm_data[lm_count * 10 + 7] = lm_data[lm_y_index];
                    pred_lm_data[lm_count * 10 + 8] = lm_data[rm_x_index];
                    pred_lm_data[lm_count * 10 + 9] = lm_data[rm_y_index];

                    lm_count++;
                }
            }
        }
    }
}
template void GetLocTruthAndPrediction(float* gt_loc_offest_data, float* pred_loc_offest_data,
                                float* gt_loc_wh_data, float* pred_loc_wh_data,
                                float* gt_lm_data, float* pred_lm_data,
                                const int output_width, const int output_height, 
                                bool share_location, const float* loc_data,
                                const float* wh_data, const float* lm_data,
                                const int loc_channels, const int lm_channels,
                                std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                                bool has_lm);
template void GetLocTruthAndPrediction(double* gt_loc_offest_data, double* pred_loc_offest_data,
                                double* gt_loc_wh_data, double* pred_loc_wh_data,
                                double* gt_lm_data, double* pred_lm_data,
                                const int output_width, const int output_height, 
                                bool share_location, const double* loc_data,
                                const double* wh_data, const double* lm_data,
                                const int loc_channels, const int lm_channels,
                                std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                                bool has_lm);                              

template <typename Dtype>
void CopySingleDiffToBottom(const Dtype* pre_offset_diff, const Dtype* pre_wh_diff, const int output_width, 
                                const int output_height, bool has_lm, const Dtype* lm_pre_diff,
                                bool share_location,  Dtype* loc_diff,
                                Dtype* wh_diff, Dtype* lm_diff,
                                const int loc_channels, const int lm_channels,
                                std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes){
    std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > ::iterator iter;
    int count = 0;
    int lm_count = 0;
    CHECK_EQ(share_location, true);
    int dimScale = output_height * output_width;
    if(has_lm){
        CHECK_EQ(loc_channels, 2);
        CHECK_EQ(lm_channels, 10);
    }else{
        CHECK_EQ(loc_channels, 2);
        CHECK_EQ(lm_channels, 1);
    }

    for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
        int batch_id = iter->first;
        vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > gt_bboxes = iter->second;
        for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
            const Dtype xmin = gt_bboxes[ii].first.xmin() * output_width;
            const Dtype ymin = gt_bboxes[ii].first.ymin() * output_height;
            const Dtype xmax = gt_bboxes[ii].first.xmax() * output_width;
            const Dtype ymax = gt_bboxes[ii].first.ymax() * output_height;
            Dtype center_x = Dtype((xmin + xmax) / 2);
            Dtype center_y = Dtype((ymin + ymax) / 2);
            int inter_center_x = static_cast<int> (center_x);
            int inter_center_y = static_cast<int> (center_y);
            int x_loc_index = batch_id * loc_channels * dimScale
                                    + 0 * dimScale + inter_center_y * output_width + inter_center_x;
            int y_loc_index = batch_id * loc_channels * dimScale 
                                    + 1 * dimScale + inter_center_y * output_width + inter_center_x;
            int width_loc_index = batch_id * loc_channels * dimScale
                                    + 0 * dimScale + inter_center_y * output_width + inter_center_x;
            int height_loc_index = batch_id * loc_channels * dimScale 
                                    + 1 * dimScale + inter_center_y * output_width + inter_center_x;
            loc_diff[x_loc_index] = pre_offset_diff[count * 2 + 0];
            loc_diff[y_loc_index] = pre_offset_diff[count * 2 + 1];
            wh_diff[width_loc_index] = pre_wh_diff[count * 2 + 0];
            wh_diff[height_loc_index] = pre_wh_diff[count * 2 + 1];
            ++count;
            if(has_lm){
                //lm_gt_datas, & lm_pred_datas
                if(gt_bboxes[ii].second.lefteye().x() > 0 && gt_bboxes[ii].second.lefteye().y() > 0 &&
                   gt_bboxes[ii].second.righteye().x() > 0 && gt_bboxes[ii].second.righteye().y() > 0 && 
                   gt_bboxes[ii].second.nose().x() > 0 && gt_bboxes[ii].second.nose().y() > 0 &&
                   gt_bboxes[ii].second.leftmouth().x() > 0 && gt_bboxes[ii].second.leftmouth().y() > 0 &&
                   gt_bboxes[ii].second.rightmouth().x() > 0 && gt_bboxes[ii].second.rightmouth().y() > 0){
                    
                    int le_x_index = batch_id * lm_channels * dimScale
                                        + 0 * dimScale + inter_center_y * output_width + inter_center_x;
                    int le_y_index = batch_id * lm_channels * dimScale 
                                        + 1 * dimScale + inter_center_y * output_width + inter_center_x;
                    int re_x_index = batch_id * lm_channels * dimScale
                                        + 2 * dimScale + inter_center_y * output_width + inter_center_x;
                    int re_y_index = batch_id * lm_channels * dimScale 
                                        + 3 * dimScale + inter_center_y * output_width + inter_center_x;
                    int no_x_index = batch_id * lm_channels * dimScale
                                        + 4 * dimScale + inter_center_y * output_width + inter_center_x;
                    int no_y_index = batch_id * lm_channels * dimScale 
                                        + 5 * dimScale + inter_center_y * output_width + inter_center_x;
                    int lm_x_index = batch_id * lm_channels * dimScale
                                        + 6 * dimScale + inter_center_y * output_width + inter_center_x;
                    int lm_y_index = batch_id * lm_channels * dimScale 
                                        + 7 * dimScale + inter_center_y * output_width + inter_center_x;
                    int rm_x_index = batch_id * lm_channels * dimScale
                                        + 8 * dimScale + inter_center_y * output_width + inter_center_x;
                    int rm_y_index = batch_id * lm_channels * dimScale 
                                        + 9 * dimScale + inter_center_y * output_width + inter_center_x;

                    lm_diff[le_x_index] = lm_pre_diff[lm_count * 10 + 0];
                    lm_diff[le_y_index] = lm_pre_diff[lm_count * 10 + 1];
                    lm_diff[re_x_index] = lm_pre_diff[lm_count * 10 + 2];
                    lm_diff[re_y_index] = lm_pre_diff[lm_count * 10 + 3];
                    lm_diff[no_x_index] = lm_pre_diff[lm_count * 10 + 4];
                    lm_diff[no_y_index] = lm_pre_diff[lm_count * 10 + 5];
                    lm_diff[lm_x_index] = lm_pre_diff[lm_count * 10 + 6];
                    lm_diff[lm_y_index] = lm_pre_diff[lm_count * 10 + 7];
                    lm_diff[rm_x_index] = lm_pre_diff[lm_count * 10 + 8];
                    lm_diff[rm_y_index] = lm_pre_diff[lm_count * 10 + 9];
                    lm_count++;
                }
            }
        }
    }
}
template void CopySingleDiffToBottom(const float* pre_offset_diff, const float* pre_wh_diff, const int output_width, 
                                const int output_height, bool has_lm, const float* lm_pre_diff,
                                bool share_location, float* loc_diff,
                                float* wh_diff, float* lm_diff,
                                const int loc_channels, const int lm_channels,
                                std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes);
template void CopySingleDiffToBottom(const double* pre_offset_diff, const double* pre_wh_diff,const int output_width, 
                                const int output_height, bool has_lm, const double* lm_pre_diff,
                                bool share_location, double* loc_diff,
                                double* wh_diff, double* lm_diff,
                                const int loc_channels, const int lm_channels,
                                std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes);
}// namespace caffe
