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
        CHECK_EQ(lm_channels, 0);
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
        CHECK_EQ(lm_channels, 0);
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


template <typename Dtype>
void get_single_topK(const Dtype* keep_max_data, const Dtype* loc_data,
                    const Dtype* wh_data, const Dtype* lm_data,
                    const int loc_channels, const int lm_channels, 
                    const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > >* results
                  , const int loc_channels, bool has_lm,  Dtype conf_thresh, Dtype nms_thresh){
    std::vector<CenterNetInfo > batch_result;
    int dim = classes * output_width * output_height;
    int dimScale = output_width * output_height;
    if(has_lm){
        CHECK_EQ(loc_channels, 2);
        CHECK_EQ(lm_channels, 10);
    }else{
        CHECK_EQ(loc_channels, 2);
        CHECK_EQ(lm_channels, 0);
    }
    for(int i = 0; i < num_batch; i++){
        std::vector<CenterNetInfo > batch_temp;
        batch_result.clear();
        for(int c = 0 ; c < classes; c++){
            for(int h = 0; h < output_height; h++){
                for(int w = 0; w < output_width; w++){
                    int index = i * dim + c * dimScale + h * output_width + w;
                    if(keep_max_data[index] > conf_thresh && keep_max_data[index] < 1){
                        int x_index = i * loc_channels * dimScale + 0 * dimScale + h * output_width + w;
                        int y_index = i * loc_channels * dimScale + 1 * dimScale + h * output_width + w;
                        int w_index = i * loc_channels * dimScale + 0 * dimScale + h * output_width + w;
                        int h_index = i * loc_channels * dimScale + 1 * dimScale + h * output_width + w;
                        Dtype center_x = (w + loc_data[x_index]) * 4;
                        Dtype center_y = (h + loc_data[y_index]) * 4;
                        Dtype width = std::exp(wh_data[w_index]) * 4 ;
                        Dtype height = std::exp(wh_data[h_index]) * 4 ;
                        Dtype xmin = GET_VALID_VALUE((center_x - Dtype(width / 2)), Dtype(0.f), Dtype(4 * output_width));
                        Dtype xmax = GET_VALID_VALUE((center_x + Dtype(width / 2)), Dtype(0.f), Dtype(4 * output_width));
                        Dtype ymin = GET_VALID_VALUE((center_y - Dtype(height / 2)), Dtype(0.f), Dtype(4 * output_height));
                        Dtype ymax = GET_VALID_VALUE((center_y + Dtype(height / 2)), Dtype(0.f), Dtype(4 * output_height));

                        CenterNetInfo temp_result;
                        temp_result.set_class_id(c);
                        temp_result.set_score(keep_max_data[index]);
                        temp_result.set_xmin(xmin);
                        temp_result.set_xmax(xmax);
                        temp_result.set_ymin(ymin);
                        temp_result.set_ymax(ymax);
                        temp_result.set_area(width * height);

                        if(has_lm){
                            int le_x_index =  i * lm_channels * dimScale + 0 * dimScale + h * output_width + w;
                            int le_y_index =  i * lm_channels * dimScale + 1 * dimScale + h * output_width + w;
                            int re_x_index =  i * lm_channels * dimScale + 2 * dimScale + h * output_width + w;
                            int re_y_index =  i * lm_channels * dimScale + 3 * dimScale + h * output_width + w;
                            int no_x_index =  i * lm_channels * dimScale + 4 * dimScale + h * output_width + w;
                            int no_y_index =  i * lm_channels * dimScale + 5 * dimScale + h * output_width + w;
                            int lm_x_index =  i * lm_channels * dimScale + 6 * dimScale + h * output_width + w;
                            int lm_y_index =  i * lm_channels * dimScale + 7 * dimScale + h * output_width + w;
                            int rm_x_index =  i * lm_channels * dimScale + 8 * dimScale + h * output_width + w;
                            int rm_y_index =  i * lm_channels * dimScale + 9 * dimScale + h * output_width + w;

                            Dtype bbox_width = xmax - xmin;
                            Dtype bbox_height = ymax - ymin;

                            Dtype le_x = GET_VALID_VALUE((center_x + loc_data[le_x_index] * bbox_width) * 4, Dtype(0.f), Dtype(4 * output_width));
                            Dtype le_y = GET_VALID_VALUE((center_y + loc_data[le_y_index] * bbox_height) * 4, Dtype(0.f),Dtype(4 * output_height));
                            Dtype re_x = GET_VALID_VALUE((center_x + loc_data[re_x_index] * bbox_width) * 4, Dtype(0.f), Dtype(4 * output_width));
                            Dtype re_y = GET_VALID_VALUE((center_y + loc_data[re_y_index] * bbox_height) * 4, Dtype(0.f),Dtype(4 * output_height));
                            Dtype no_x = GET_VALID_VALUE((center_x + loc_data[no_x_index] * bbox_width) * 4, Dtype(0.f), Dtype(4 * output_width));
                            Dtype no_y = GET_VALID_VALUE((center_y + loc_data[no_y_index] * bbox_height) * 4, Dtype(0.f),Dtype(4 * output_height));
                            Dtype lm_x = GET_VALID_VALUE((center_x + loc_data[lm_x_index] * bbox_width) * 4, Dtype(0.f), Dtype(4 * output_width));
                            Dtype lm_y = GET_VALID_VALUE((center_y + loc_data[lm_y_index] * bbox_height) * 4, Dtype(0.f),Dtype(4 * output_height));
                            Dtype rm_x = GET_VALID_VALUE((center_x + loc_data[rm_x_index] * bbox_width) * 4, Dtype(0.f), Dtype(4 * output_width));
                            Dtype rm_y = GET_VALID_VALUE((center_y + loc_data[rm_y_index] * bbox_height) * 4, Dtype(0.f),Dtype(4 * output_height));

                            temp_result.mutable_marks()->mutable_lefteye()->set_x(le_x);
                            temp_result.mutable_marks()->mutable_lefteye()->set_y(le_y);
                            temp_result.mutable_marks()->mutable_righteye()->set_x(re_x);
                            temp_result.mutable_marks()->mutable_righteye()->set_y(re_y);
                            temp_result.mutable_marks()->mutable_nose()->set_x(no_x);
                            temp_result.mutable_marks()->mutable_nose()->set_y(no_y);
                            temp_result.mutable_marks()->mutable_leftmouth()->set_x(lm_x);
                            temp_result.mutable_marks()->mutable_leftmouth()->set_y(lm_y);
                            temp_result.mutable_marks()->mutable_rightmouth()->set_x(rm_x);
                            temp_result.mutable_marks()->mutable_rightmouth()->set_y(rm_y);
                        }
                        batch_temp.push_back(temp_result);
                    } 
                }
            }
        }
        hard_nms(batch_temp, &batch_result, nms_thresh);
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
template  void get_single_topK(const float* keep_max_data, const float* loc_data, 
                    const float* wh_data, const float* lm_data,
                    const int loc_channels, const int lm_channels, 
                    const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > >* results
                  , const int loc_channels, bool has_lm, float conf_thresh, float nms_thresh);
template void get_single_topK(const double* keep_max_data, const double* loc_data, 
                    const double* wh_data, const double* lm_data,
                    const int loc_channels, const int lm_channels, 
                    const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > >* results
                  , const int loc_channels, bool has_lm, double conf_thresh, double nms_thresh);



}  // namespace caffe
