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
#include "caffe/util/center_util.hpp"
#include "caffe/util/center_bbox_util.hpp"

#define GET_VALID_VALUE(value, min, max) ((((value) >= (min) ? (value) : (min)) < (max) ? ((value) >= (min) ? (value) : (min)): (max)))

#define FOCAL_LOSS_SOFTMAX true 

int count_gt = 0;
int count_one = 0;
namespace caffe {
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


void hard_nms(std::vector<CenterNetInfo>& input, std::vector<CenterNetInfo>* output, float nmsthreshold,int type){
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


void soft_nms(std::vector<CenterNetInfo>& input, std::vector<CenterNetInfo>* output, 
                        float sigma, float Nt, 
                        float threshold, unsigned int type){
    unsigned int numBoxes = input.size();
    float iw, ih;
    float ua;
    int pos = 0;
    float maxscore = 0;
    int maxpos = 0;
    float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov;

    for(int ii = 0; ii < numBoxes; ii++){
        maxscore = input[ii].score();
        maxpos = ii;
        tx1 = input[ii].xmin();
        ty1 = input[ii].ymin();
        tx2 = input[ii].xmax();
        ty2 = input[ii].ymax();
        ts = input[ii].score();

        pos = ii + 1;

        while(pos < numBoxes){
            if(maxscore < input[pos].score()){
                maxscore = input[pos].score();
                maxpos = pos;
            }
            pos = pos + 1;
        }

        // add max box as a detection 

        input[ii].set_xmin(input[maxpos].xmin());
        input[ii].set_xmax(input[maxpos].xmax());
        input[ii].set_ymin(input[maxpos].ymin());
        input[ii].set_ymax(input[maxpos].ymax());
        input[ii].set_score(input[maxpos].score());

        input[maxpos].set_xmin(tx1);
        input[maxpos].set_xmax(tx2);
        input[maxpos].set_ymin(ty1);
        input[maxpos].set_ymax(ty2);
        input[maxpos].set_score(ts);

        tx1 = input[ii].xmin();
        ty1 = input[ii].ymin();
        tx2 = input[ii].xmax();
        ty2 = input[ii].ymax();
        ts = input[ii].score();

        pos = ii + 1;
        // NMS iterations, note that N changes if detection boxes fall below threshold
        while(pos < numBoxes){
            x1 = input[pos].xmin();
            y1 = input[pos].ymin();
            x2 = input[pos].xmax();
            y2 = input[pos].ymax();
            //s = input[pos].score();
            area = (x2 - x1 + 1) * (y2 - y1 + 1);
            iw = (std::min(tx2, x2) - std::max(tx1, x1) + 1);
            if(iw > 0){
                ih = (std::min(ty2, y2) - std::max(ty1, y1) + 1);
                if(ih > 0){
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih);
                    ov = (float) iw * ih / ua  ;//  iou between max box and detection box

                    if(type == 1) {// linear
                        if(ov > Nt)
                            weight = 1 - ov;
                        else
                            weight = 1;
                    }
                    else if(type == 2) // gaussian
                        weight = std::exp( -1 * (float)(ov * ov)/sigma);
                    else{
                        if(ov > Nt) 
                            weight = 0;
                        else
                            weight = 1;
                    }
                    input[pos].set_score(weight * input[pos].score()); 
                    //if box score falls below threshold, discard the box by swapping with last box
                    //update numBoxes
                    if(input[pos].score() < threshold){
                        input[pos].set_xmin(input[numBoxes - 1].xmin());
                        input[pos].set_xmax(input[numBoxes - 1].xmax());
                        input[pos].set_ymin(input[numBoxes - 1].ymin());
                        input[pos].set_ymax(input[numBoxes - 1].ymax());
                        input[pos].set_score(input[numBoxes - 1].score());
                        numBoxes = numBoxes - 1;
                        pos = pos - 1;
                    }
                }
            }
            pos = pos + 1;
        }
    }
    for(int ii = 0; ii < numBoxes; ii++){
        output->push_back(input[ii]);
    }
}


template <typename Dtype>
void EncodeTruthAndPredictions(Dtype* gt_loc_offest_data, Dtype* pred_loc_offest_data,
                                Dtype* gt_loc_wh_data, Dtype* pred_loc_wh_data,
                                Dtype* gt_lm_data, Dtype* pred_lm_data,
                                const int output_width, const int output_height, 
                                bool share_location, const Dtype* channel_loc_data,
                                const int num_channels, std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                                bool has_lm){
    std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > ::iterator iter;
    CHECK_EQ(share_location, true);
    int dimScale = output_height * output_width;
    int count = 0;
    int lm_count = 0;
    if(has_lm){
        CHECK_EQ(num_channels, 14);
    }else{
        CHECK_EQ(num_channels, 4);
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

            int x_index = batch_id * num_channels * dimScale
                                    + 0 * dimScale
                                    + inter_center_y * output_width + inter_center_x;
            int y_index = batch_id * num_channels * dimScale 
                                    + 1 * dimScale
                                    + inter_center_y * output_width + inter_center_x;
            int width_index = batch_id * num_channels * dimScale
                                    + 2 * dimScale
                                    + inter_center_y * output_width + inter_center_x;
            int height_index = batch_id * num_channels * dimScale 
                                    + 3 * dimScale
                                    + inter_center_y * output_width + inter_center_x;
            gt_loc_offest_data[count * 2 + 0] = diff_x;
            gt_loc_offest_data[count * 2 + 1] = diff_y;
            gt_loc_wh_data[count * 2 + 0] = std::log(width);
            gt_loc_wh_data[count * 2 + 1] = std::log(height);
            pred_loc_offest_data[count * 2 + 0] = channel_loc_data[x_index];
            pred_loc_offest_data[count * 2 + 1] = channel_loc_data[y_index];
            pred_loc_wh_data[count * 2 + 0] = channel_loc_data[width_index];
            pred_loc_wh_data[count * 2 + 1] = channel_loc_data[height_index];
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

                    int le_x_index = batch_id * num_channels * dimScale
                                        + 4 * dimScale + inter_center_y * output_width + inter_center_x;
                    int le_y_index = batch_id * num_channels * dimScale 
                                        + 5 * dimScale + inter_center_y * output_width + inter_center_x;
                    int re_x_index = batch_id * num_channels * dimScale
                                        + 6 * dimScale + inter_center_y * output_width + inter_center_x;
                    int re_y_index = batch_id * num_channels * dimScale 
                                        + 7 * dimScale + inter_center_y * output_width + inter_center_x;
                    int no_x_index = batch_id * num_channels * dimScale
                                        + 8 * dimScale + inter_center_y * output_width + inter_center_x;
                    int no_y_index = batch_id * num_channels * dimScale 
                                        + 9 * dimScale + inter_center_y * output_width + inter_center_x;
                    int lm_x_index = batch_id * num_channels * dimScale
                                        + 10 * dimScale + inter_center_y * output_width + inter_center_x;
                    int lm_y_index = batch_id * num_channels * dimScale 
                                        + 11 * dimScale + inter_center_y * output_width + inter_center_x;
                    int rm_x_index = batch_id * num_channels * dimScale
                                        + 12 * dimScale + inter_center_y * output_width + inter_center_x;
                    int rm_y_index = batch_id * num_channels * dimScale 
                                        + 13 * dimScale + inter_center_y * output_width + inter_center_x;

                    pred_lm_data[lm_count * 10 + 0] = channel_loc_data[le_x_index];
                    pred_lm_data[lm_count * 10 + 1] = channel_loc_data[le_y_index];
                    pred_lm_data[lm_count * 10 + 2] = channel_loc_data[re_x_index];
                    pred_lm_data[lm_count * 10 + 3] = channel_loc_data[re_y_index];
                    pred_lm_data[lm_count * 10 + 4] = channel_loc_data[no_x_index];
                    pred_lm_data[lm_count * 10 + 5] = channel_loc_data[no_y_index];
                    pred_lm_data[lm_count * 10 + 6] = channel_loc_data[lm_x_index];
                    pred_lm_data[lm_count * 10 + 7] = channel_loc_data[lm_y_index];
                    pred_lm_data[lm_count * 10 + 8] = channel_loc_data[rm_x_index];
                    pred_lm_data[lm_count * 10 + 9] = channel_loc_data[rm_y_index];

                    ++lm_count;
                }
            }
        }
    }
}
template void EncodeTruthAndPredictions(float* gt_loc_offest_data, float* pred_loc_offest_data,
                                float* gt_loc_wh_data, float* pred_loc_wh_data,
                                float* gt_lm_data, float* pred_lm_data,
                                const int output_width, const int output_height, 
                                bool share_location, const float* channel_loc_data,
                                const int num_channels, std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes, 
                                bool has_lm);
template void EncodeTruthAndPredictions(double* gt_loc_offest_data, double* pred_loc_offest_data,
                                double* gt_loc_wh_data, double* pred_loc_wh_data,
                                double* gt_lm_data, double* pred_lm_data,
                                const int output_width, const int output_height, 
                                bool share_location, const double* channel_loc_data,
                                const int num_channels, std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                                bool has_lm);                              

template <typename Dtype>
void CopyDiffToBottom(const Dtype* pre_offset_diff, const Dtype* pre_wh_diff, const int output_width, 
                                const int output_height, bool has_lm, const Dtype* lm_pre_diff,
                                bool share_location, Dtype* bottom_diff, const int num_channels,
                                std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes){
    std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > ::iterator iter;
    int count = 0;
    int lm_count = 0;
    CHECK_EQ(share_location, true);
    int dimScale = output_height * output_width;
    if(has_lm){
        CHECK_EQ(num_channels, 14);
    }else{
        CHECK_EQ(num_channels, 4);
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
            int x_index = batch_id * num_channels * dimScale
                                    + 0 * dimScale + inter_center_y * output_width + inter_center_x;
            int y_index = batch_id * num_channels * dimScale 
                                    + 1 * dimScale + inter_center_y * output_width + inter_center_x;
            int width_index = batch_id * num_channels * dimScale
                                    + 2 * dimScale + inter_center_y * output_width + inter_center_x;
            int height_index = batch_id * num_channels * dimScale 
                                    + 3 * dimScale + inter_center_y * output_width + inter_center_x;
            bottom_diff[x_index] = pre_offset_diff[count * 2 + 0];
            bottom_diff[y_index] = pre_offset_diff[count * 2 + 1];
            bottom_diff[width_index] = pre_wh_diff[count * 2 + 0];
            bottom_diff[height_index] = pre_wh_diff[count * 2 + 1];
            ++count;
            if(has_lm){
                //lm_gt_datas, & lm_pred_datas
                if(gt_bboxes[ii].second.lefteye().x() > 0 && gt_bboxes[ii].second.lefteye().y() > 0 &&
                   gt_bboxes[ii].second.righteye().x() > 0 && gt_bboxes[ii].second.righteye().y() > 0 && 
                   gt_bboxes[ii].second.nose().x() > 0 && gt_bboxes[ii].second.nose().y() > 0 &&
                   gt_bboxes[ii].second.leftmouth().x() > 0 && gt_bboxes[ii].second.leftmouth().y() > 0 &&
                   gt_bboxes[ii].second.rightmouth().x() > 0 && gt_bboxes[ii].second.rightmouth().y() > 0){
                    
                    int le_x_index = batch_id * num_channels * dimScale
                                        + 4 * dimScale + inter_center_y * output_width + inter_center_x;
                    int le_y_index = batch_id * num_channels * dimScale 
                                        + 5 * dimScale + inter_center_y * output_width + inter_center_x;
                    int re_x_index = batch_id * num_channels * dimScale
                                        + 6 * dimScale + inter_center_y * output_width + inter_center_x;
                    int re_y_index = batch_id * num_channels * dimScale 
                                        + 7 * dimScale + inter_center_y * output_width + inter_center_x;
                    int no_x_index = batch_id * num_channels * dimScale
                                        + 8 * dimScale + inter_center_y * output_width + inter_center_x;
                    int no_y_index = batch_id * num_channels * dimScale 
                                        + 9 * dimScale + inter_center_y * output_width + inter_center_x;
                    int lm_x_index = batch_id * num_channels * dimScale
                                        + 10 * dimScale + inter_center_y * output_width + inter_center_x;
                    int lm_y_index = batch_id * num_channels * dimScale 
                                        + 11 * dimScale + inter_center_y * output_width + inter_center_x;
                    int rm_x_index = batch_id * num_channels * dimScale
                                        + 12 * dimScale + inter_center_y * output_width + inter_center_x;
                    int rm_y_index = batch_id * num_channels * dimScale 
                                        + 13 * dimScale + inter_center_y * output_width + inter_center_x;

                    bottom_diff[le_x_index] = lm_pre_diff[lm_count * 10 + 0];
                    bottom_diff[le_y_index] = lm_pre_diff[lm_count * 10 + 1];
                    bottom_diff[re_x_index] = lm_pre_diff[lm_count * 10 + 2];
                    bottom_diff[re_y_index] = lm_pre_diff[lm_count * 10 + 3];
                    bottom_diff[no_x_index] = lm_pre_diff[lm_count * 10 + 4];
                    bottom_diff[no_y_index] = lm_pre_diff[lm_count * 10 + 5];
                    bottom_diff[lm_x_index] = lm_pre_diff[lm_count * 10 + 6];
                    bottom_diff[lm_y_index] = lm_pre_diff[lm_count * 10 + 7];
                    bottom_diff[rm_x_index] = lm_pre_diff[lm_count * 10 + 8];
                    bottom_diff[rm_y_index] = lm_pre_diff[lm_count * 10 + 9];
                    ++lm_count;
                }
            }
        }
    }
}
template void CopyDiffToBottom(const float* pre_offset_diff, const float* pre_wh_diff, const int output_width, 
                                const int output_height, bool has_lm, const float* lm_pre_diff,
                                bool share_location, float* bottom_diff, const int num_channels,
                                std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes);
template void CopyDiffToBottom(const double* pre_offset_diff, const double* pre_wh_diff,const int output_width, 
                                const int output_height, bool has_lm, const double* lm_pre_diff,
                                bool share_location, double* bottom_diff, const int num_channels,
                                std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes);


template <typename Dtype>
void get_topK(const Dtype* keep_max_data, const Dtype* loc_data, const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > >* results
                  , const int loc_channels, bool has_lm,  Dtype conf_thresh, Dtype nms_thresh){
    std::vector<CenterNetInfo > batch_result;
    int dim = classes * output_width * output_height;
    int dimScale = output_width * output_height;
    if(has_lm){
        CHECK_EQ(loc_channels, 14);
    }else{
        CHECK_EQ(loc_channels, 4);
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
                        int w_index = i * loc_channels * dimScale + 2 * dimScale + h * output_width + w;
                        int h_index = i * loc_channels * dimScale + 3 * dimScale + h * output_width + w;
                        Dtype center_x = (w + loc_data[x_index]) * 4;
                        Dtype center_y = (h + loc_data[y_index]) * 4;
                        Dtype width = std::exp(loc_data[w_index]) * 4 ;
                        Dtype height = std::exp(loc_data[h_index]) * 4 ;
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
                            int le_x_index =  i * loc_channels * dimScale + 4 * dimScale + h * output_width + w;
                            int le_y_index =  i * loc_channels * dimScale + 5 * dimScale + h * output_width + w;
                            int re_x_index =  i * loc_channels * dimScale + 6 * dimScale + h * output_width + w;
                            int re_y_index =  i * loc_channels * dimScale + 7 * dimScale + h * output_width + w;
                            int no_x_index =  i * loc_channels * dimScale + 8 * dimScale + h * output_width + w;
                            int no_y_index =  i * loc_channels * dimScale + 9 * dimScale + h * output_width + w;
                            int lm_x_index =  i * loc_channels * dimScale + 10 * dimScale + h * output_width + w;
                            int lm_y_index =  i * loc_channels * dimScale + 11 * dimScale + h * output_width + w;
                            int rm_x_index =  i * loc_channels * dimScale + 12 * dimScale + h * output_width + w;
                            int rm_y_index =  i * loc_channels * dimScale + 13 * dimScale + h * output_width + w;

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
template  void get_topK(const float* keep_max_data, const float* loc_data, const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > >* results
                  , const int loc_channels, bool has_lm, float conf_thresh, float nms_thresh);
template void get_topK(const double* keep_max_data, const double* loc_data, const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > >* results
                  , const int loc_channels, bool has_lm, double conf_thresh, double nms_thresh);



template <typename Dtype>
void GenerateBatchHeatmap(std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes, 
                            Dtype* gt_heatmap, 
                            const int num_classes_, const int output_width, const int output_height){
    std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > ::iterator iter;
    count_gt = 0;

    for(iter = all_gt_bboxes.begin(); iter != all_gt_bboxes.end(); iter++){
        int batch_id = iter->first;
        vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > gt_bboxes = iter->second;
        for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
            std::vector<Dtype> heatmap((output_width *output_height), Dtype(0.));
            const int class_id = gt_bboxes[ii].first.label();
            Dtype *classid_heap = gt_heatmap + (batch_id * num_classes_ + (class_id - 1)) * output_width * output_height;
            const Dtype xmin = gt_bboxes[ii].first.xmin() * output_width;
            const Dtype ymin = gt_bboxes[ii].first.ymin() * output_height;
            const Dtype xmax = gt_bboxes[ii].first.xmax() * output_width;
            const Dtype ymax = gt_bboxes[ii].first.ymax() * output_height;
            const Dtype width = Dtype(xmax - xmin);
            const Dtype height = Dtype(ymax - ymin);
            Dtype radius = gaussian_radius(height, width, Dtype(0.7));
            radius = std::max(0, int(radius));
            int center_x = static_cast<int>(Dtype((xmin + xmax) / 2));
            int center_y = static_cast<int>(Dtype((ymin + ymax) / 2));
            draw_umich_gaussian( heatmap, center_x, center_y, radius, output_height, output_width );
            transferCVMatToBlobData(heatmap, classid_heap);
            count_gt++;
        }
    }
}
template void GenerateBatchHeatmap(std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes, float* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);
template void GenerateBatchHeatmap(std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes, double* gt_heatmap, 
                              const int num_classes_, const int output_width, const int output_height);

// 置信度得分,用逻辑回归来做,loss_delta梯度值,既前向又后向
template <typename Dtype>
void EncodeYoloObject(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int net_width, const int net_height,
                          Dtype* channel_pred_data,
                          std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                          std::vector<int> mask_bias, std::vector<std::pair<Dtype, Dtype> >bias_scale, 
                          Dtype* bottom_diff, Dtype ignore_thresh, YoloScoreShow *Score){
    CHECK_EQ(net_height, net_width);
    int stride_channel = 4 + 1 + num_classes;
    //int stride_feature = net_height / output_height;
    int dimScale = output_height * output_width;
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    CHECK_EQ(num_channels, (5 + num_classes) * mask_bias.size()) 
            << "num_channels shoule be set to including bias_x, bias_y, width, height, object_confidence and classes";
    for(int b = 0; b < batch_size; b++){
        for(unsigned m = 0; m < mask_bias.size(); m++){
            int x_index = b * num_channels * dimScale
                                        + (m * stride_channel + 0)* dimScale;
            for(int i = 0; i < 2 * dimScale; i++)
                channel_pred_data[x_index + i] = CenterSigmoid(channel_pred_data[x_index + i]);
            int object_index = b * num_channels * dimScale
                                        + (m * stride_channel + 4)* dimScale;
            for(int i = 0; i < (num_classes + 1) * dimScale; i++){
                channel_pred_data[object_index + i] = CenterSigmoid(channel_pred_data[object_index + i]);
            }
        }
    }
    for(int b = 0; b < batch_size; b++){
        vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > gt_bboxes = all_gt_bboxes.find(b)->second;
        for(int h = 0; h < output_height; h++){
            for(int w = 0; w < output_width; w++){
                for(unsigned m = 0; m < mask_bias.size(); m++){
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
                    float bb_center_x = (channel_pred_data[x_index] + w) / output_width;
                    float bb_center_y = (channel_pred_data[y_index] + h) / output_height;
                    float bb_width = (float)std::exp(channel_pred_data[width_index]) 
                                                                    * bias_scale[mask_bias[m]].first / net_width;
                    float bb_height = (float)std::exp(channel_pred_data[height_index]) 
                                                                    * bias_scale[mask_bias[m]].second / net_height;
                    predBox.set_xmin(bb_center_x - bb_width / 2);
                    predBox.set_xmax(bb_center_x + bb_width / 2);
                    predBox.set_ymin(bb_center_y - bb_height / 2);
                    predBox.set_ymax(bb_center_y + bb_height / 2);
                    float best_iou = 0;
                    for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
                        float iou = YoloBBoxIou(predBox, gt_bboxes[ii].first);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    avg_anyobj += channel_pred_data[object_index];
                    bottom_diff[object_index] = (-1) * (0 - channel_pred_data[object_index]);
                    if(best_iou > ignore_thresh){
                        bottom_diff[object_index] = 0;
                    }
                }
            }
        }
        for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
            const Dtype xmin = gt_bboxes[ii].first.xmin();
            const Dtype ymin = gt_bboxes[ii].first.ymin();
            const Dtype xmax = gt_bboxes[ii].first.xmax();
            const Dtype ymax = gt_bboxes[ii].first.ymax();
            float best_iou = 0.f;
            int best_mask_scale = 0;
            for(unsigned m = 0; m < bias_scale.size(); m++){
                NormalizedBBox anchor_bbox ; 
                NormalizedBBox shfit_gt_bbox;
                Dtype shift_x_min = 0 - Dtype((xmax - xmin) / 2);
                Dtype shift_x_max = 0 + Dtype((xmax - xmin) / 2);
                Dtype shift_y_min = 0 - Dtype((ymax - ymin) / 2);
                Dtype shift_y_max = 0 + Dtype((ymax - ymin) / 2);
                shfit_gt_bbox.set_xmin(shift_x_min);
                shfit_gt_bbox.set_xmax(shift_x_max);
                shfit_gt_bbox.set_ymin(shift_y_min);
                shfit_gt_bbox.set_ymax(shift_y_max);
                Dtype bias_width = bias_scale[m].first;
                Dtype bias_height = bias_scale[m].second;
                float bias_xmin = 0 - (float)bias_width / (2 * net_width);
                float bias_ymin = 0 - (float)bias_height / (2 * net_height);
                float bias_xmax = 0 + (float)bias_width / (2 * net_width);
                float bias_ymax = 0 + (float)bias_height / (2 * net_height);
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
            if(mask_n >= 0){
                Dtype center_x = Dtype((xmin + xmax) / 2) * output_width;
                Dtype center_y = Dtype((ymin + ymax) / 2) * output_height;
                int inter_center_x = static_cast<int> (center_x);
                int inter_center_y = static_cast<int> (center_y);
                Dtype diff_x = center_x - inter_center_x;
                Dtype diff_y = center_y - inter_center_y;
                Dtype width = std::log((xmax - xmin) * net_width / bias_scale[best_mask_scale].first);
                Dtype height = std::log((ymax - ymin) * net_height / bias_scale[best_mask_scale].second);
                
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
                float delta_scale = 2 - (float)(xmax - xmin) * (ymax - ymin);
                bottom_diff[x_index] = (-1) * delta_scale * (diff_x - channel_pred_data[x_index]);
                bottom_diff[y_index] = (-1) *delta_scale * (diff_y - channel_pred_data[y_index]);
                bottom_diff[width_index] = (-1) *delta_scale * (width - channel_pred_data[width_index]);
                bottom_diff[height_index] = (-1) *delta_scale * (height - channel_pred_data[height_index]);
                bottom_diff[object_index] = (-1) * (1 - channel_pred_data[object_index]);
                avg_obj +=  channel_pred_data[object_index];
                // class score
                // 特殊情况,face数据集,包含了背景目标,而实际上不需要背景目标,所以减一
                int class_lable = gt_bboxes[ii].first.label() - 1; 
                int class_index = b * num_channels * dimScale + (mask_n * stride_channel + 5)* dimScale
                                            + inter_center_y * output_width + inter_center_x;
                if ( bottom_diff[class_index]){
                    bottom_diff[class_index + class_lable * dimScale] = 1 - channel_pred_data[class_index + class_lable * dimScale];
                    avg_cat += channel_pred_data[class_index + class_lable * dimScale];
                }else{
                    for(int c = 0; c < num_classes; c++){
                    bottom_diff[class_index + c * dimScale] =(-1) * (((c == class_lable)?1 : 0) - channel_pred_data[class_index + c * dimScale]);
                    if(c == class_lable) 
                        avg_cat += channel_pred_data[class_index + c * dimScale];
                    }
                }

                count++;
                class_count++;
                NormalizedBBox predBox;
                float bb_center_x = (channel_pred_data[x_index] + inter_center_x) / output_width;
                float bb_center_y = (channel_pred_data[y_index] + inter_center_y) / output_height;
                float bb_width = (float)std::exp(channel_pred_data[width_index]) 
                                                                * bias_scale[best_mask_scale].first / net_width;
                float bb_height = (float)std::exp(channel_pred_data[height_index]) 
                                                                * bias_scale[best_mask_scale].second / net_height;
                predBox.set_xmin(bb_center_x - bb_width / 2);
                predBox.set_xmax(bb_center_x + bb_width / 2);
                predBox.set_ymin(bb_center_y - bb_height / 2);
                predBox.set_ymax(bb_center_y + bb_height / 2);
                float iou = YoloBBoxIou(predBox, gt_bboxes[ii].first);
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
                
            }
        } 
    }
    Score->avg_anyobj = avg_anyobj;
    Score->avg_cat = avg_cat;
    Score->avg_iou = avg_iou;
    Score->avg_obj = avg_obj;
    Score->class_count = class_count;
    Score->count = count;
    Score->recall75 =recall75;
    Score->recall = recall;
}
template void EncodeYoloObject(const int batch_size, const int num_channels, const int num_classes,
                              const int output_width, const int output_height, 
                              const int net_width, const int net_height,
                              float* channel_pred_data,
                              std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                              std::vector<int> mask_bias, std::vector<std::pair<float, float> >bias_scale, 
                              float* bottom_diff, float ignore_thresh, YoloScoreShow *Score);
template void EncodeYoloObject(const int batch_size, const int num_channels, const int num_classes,
                              const int output_width, const int output_height, 
                              const int net_width, const int net_height,
                              double* channel_pred_data,
                              std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                              std::vector<int> mask_bias, std::vector<std::pair<double, double> >bias_scale, 
                              double* bottom_diff, double ignore_thresh, YoloScoreShow *Score);



template <typename Dtype>
void GetYoloGroundTruth(const Dtype* gt_data, int num_gt,
      const int background_label_id, const bool use_difficult_gt, bool has_lm,
      std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > >* all_gt_bboxes, int batch_size){
    all_gt_bboxes->clear();

    for(int b = 0; b < batch_size; b++){
        for (int i = 0; i < num_gt; ++i) {
            int start_idx = b * num_gt * 8 +  i * 8;
            if(has_lm){
                start_idx = b * num_gt * 19 +  i * 19;
            }
            int item_id = gt_data[start_idx];
            if (item_id == -1) {
                continue;
            }
            CHECK_EQ(b ,item_id);
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
            AnnoFaceLandmarks lmarks;
            if(has_lm){
                Dtype bbox_has_lm = gt_data[start_idx + 8];
                if(bbox_has_lm > 0){
                    lmarks.mutable_lefteye()->set_x(gt_data[start_idx + 9]);
                    lmarks.mutable_lefteye()->set_y(gt_data[start_idx + 10]);
                    lmarks.mutable_righteye()->set_x(gt_data[start_idx + 11]);
                    lmarks.mutable_righteye()->set_y(gt_data[start_idx + 12]);
                    lmarks.mutable_nose()->set_x(gt_data[start_idx + 13]);
                    lmarks.mutable_nose()->set_y(gt_data[start_idx + 14]);
                    lmarks.mutable_leftmouth()->set_x(gt_data[start_idx + 15]);
                    lmarks.mutable_leftmouth()->set_y(gt_data[start_idx + 16]);
                    lmarks.mutable_rightmouth()->set_x(gt_data[start_idx + 17]);
                    lmarks.mutable_rightmouth()->set_y(gt_data[start_idx + 18]);
                }else{
                    lmarks.mutable_lefteye()->set_x(-1.);
                    lmarks.mutable_lefteye()->set_y(-1.);
                    lmarks.mutable_righteye()->set_x(-1.);
                    lmarks.mutable_righteye()->set_y(-1.);
                    lmarks.mutable_nose()->set_x(-1.);
                    lmarks.mutable_nose()->set_y(-1.);
                    lmarks.mutable_leftmouth()->set_x(-1.);
                    lmarks.mutable_leftmouth()->set_y(-1.);
                    lmarks.mutable_rightmouth()->set_x(-1.);
                    lmarks.mutable_rightmouth()->set_y(-1.);
                }
            }
            (*all_gt_bboxes)[item_id].push_back(std::make_pair(bbox, lmarks));
        }
    }
}
template void GetYoloGroundTruth(const float* gt_data, int num_gt,
      const int background_label_id, const bool use_difficult_gt, bool has_lm,
      std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > >* all_gt_bboxes, int batch_size);

template void GetYoloGroundTruth(const double* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt, bool has_lm,
      std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > >* all_gt_bboxes, int batch_size);


// 每一层使用感受野作为anchor,此anchor只匹配相对应大小范围的gt_boxes, 
// anchor 生成的方式是按照感受野的大小来生成的,所以每层只有一个感受野大小的anchor, 用于匹配相应的gt_boxes;
// anchor 的匹配方式是按照每个anchor中心落在真实框之内匹配呢？,还是直接基于每个格子中心来进行预测呢？
// loss -01: loc loss 
// loss -02: class loss 分类概率, 使用focalloss,对所有负样本进行计算
// 只针对单类物体,二分类物体检测
template <typename Dtype> 
Dtype EncodeCenterGridObjectSigmoidLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio,
                          Dtype* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                          Dtype* class_label, Dtype* bottom_diff, 
                          Dtype ignore_thresh, int *count_postive, Dtype *loc_loss_value){
    CHECK_EQ(num_classes, 1);
    int dimScale = output_height * output_width;
    Dtype score_loss = Dtype(0.), loc_loss = Dtype(0.);
    CHECK_EQ(num_channels, (4 + num_classes)) 
            << "num_channels shoule be set to including bias_x, bias_y, width, height, classes";
    for(int b = 0; b < batch_size; b++){
        int object_index = b * num_channels * dimScale + 4 * dimScale;
        for(int i = 0; i < 1 * dimScale; i++){
            channel_pred_data[object_index + i] = CenterSigmoid(channel_pred_data[object_index + i]);
        }
    }
    int postive = 0;
    caffe_set(batch_size * dimScale, Dtype(0.5f), class_label);
    for(int b = 0; b < batch_size; b++){
        vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > gt_bboxes = all_gt_bboxes.find(b)->second;
        std::vector<int> mask_Rf_anchor(dimScale, 0);
        int count = 0;
        for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
            const Dtype xmin = gt_bboxes[ii].first.xmin() * output_width;
            const Dtype ymin = gt_bboxes[ii].first.ymin() * output_height;
            const Dtype xmax = gt_bboxes[ii].first.xmax() * output_width;
            const Dtype ymax = gt_bboxes[ii].first.ymax() * output_height;
            const int gt_bbox_width = static_cast<int>((xmax - xmin) * downRatio);
            const int gt_bbox_height = static_cast<int>((ymax - ymin) * downRatio);
            for(int h = static_cast<int>(ymin); h < static_cast<int>(ymax); h++){
                for(int w = static_cast<int>(xmin); w < static_cast<int>(xmax); w++){
                    int class_index = b * dimScale +  h * output_width + w;
                    class_label[class_index] = -10;
                }
            }
            int large_side = std::max(gt_bbox_height, gt_bbox_width);
            if(large_side >= loc_truth_scale.first && large_side < loc_truth_scale.second){
                for(int h = static_cast<int>(ymin); h < static_cast<int>(ymax); h++){
                    for(int w = static_cast<int>(xmin); w < static_cast<int>(xmax); w++){
                        
                        if(mask_Rf_anchor[h * output_width + w] == 1) // 避免同一个anchor的中心落在多个gt里面
                            continue;
                        Dtype xmin_bias = (w + 0.5 - xmin) * downRatio * 2 / anchor_scale;
                        Dtype ymin_bias = (h + 0.5 - ymin) * downRatio * 2 / anchor_scale;
                        Dtype xmax_bias = (w + 0.5 - xmax) * downRatio * 2 / anchor_scale;
                        Dtype ymax_bias = (h + 0.5 - ymax) * downRatio * 2 / anchor_scale;
                        int xmin_index = b * num_channels * dimScale
                                                    + 0 * dimScale + h * output_width + w;
                        int ymin_index = b * num_channels * dimScale 
                                                    + 1 * dimScale + h * output_width + w;
                        int xmax_index = b * num_channels * dimScale
                                                    + 2 * dimScale + h * output_width + w;
                        int ymax_index = b * num_channels * dimScale 
                                                    + 3 * dimScale + h * output_width + w;
                        Dtype xmin_diff, ymin_diff, xmax_diff, ymax_diff;
                        loc_loss += smoothL1_Loss(Dtype(channel_pred_data[xmin_index] - xmin_bias), &xmin_diff);
                        loc_loss += smoothL1_Loss(Dtype(channel_pred_data[ymin_index] - ymin_bias), &ymin_diff);
                        loc_loss += smoothL1_Loss(Dtype(channel_pred_data[xmax_index] - xmax_bias), &xmax_diff);
                        loc_loss += smoothL1_Loss(Dtype(channel_pred_data[ymax_index] - ymax_bias), &ymax_diff);

                        bottom_diff[xmin_index] = xmin_diff;
                        bottom_diff[ymin_index] = ymin_diff;
                        bottom_diff[xmax_index] = xmax_diff;
                        bottom_diff[ymax_index] = ymax_diff;
                        // class score 
                        // 特殊情况,face数据集,包含了背景目标,而实际上不需要背景目标
                        int class_index = b * dimScale
                                                +  h * output_width + w;
                        class_label[class_index] = 1;
                        mask_Rf_anchor[h * output_width + w] = 1;
                        count++;
                    }
                }
            }
        }
        int gt_class_index =  b * dimScale;
        int pred_class_index = b * num_channels * dimScale + 4* dimScale;
        score_loss += FocalLossSigmoid(class_label + gt_class_index, channel_pred_data + pred_class_index, 
                                        dimScale, bottom_diff + pred_class_index);
        postive += count;
    }
    *count_postive = postive;
    *loc_loss_value = loc_loss;
    return score_loss;
}

template float EncodeCenterGridObjectSigmoidLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio,
                          float* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                          float* class_label, float* bottom_diff, 
                          float ignore_thresh, int *count_postive, float *loc_loss_value);

template double EncodeCenterGridObjectSigmoidLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio,
                          double* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                          double* class_label, double* bottom_diff, 
                          double ignore_thresh, int *count_postive, double *loc_loss_value);

template <typename Dtype>
void GetCenterGridObjectResultSigmoid(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio,
                          Dtype* channel_pred_data, const int anchor_scale, Dtype conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results){
    // face class 人脸类型
    CHECK_EQ(num_classes, 1);
    CHECK_EQ(num_channels, 4 + num_classes);
    int dimScale = output_height * output_width;
    for(int b = 0; b < batch_size; b++){
        int object_index = b * num_channels * dimScale + 4 * dimScale;
        for(int i = 0; i < num_classes * dimScale; i++){
            channel_pred_data[object_index + i] = CenterSigmoid(channel_pred_data[object_index + i]);
        }
    }

    for(int b = 0; b < batch_size; b++){
        for(int h = 0; h < output_height; h++){
            for(int w = 0; w < output_width; w++){
                int x_index = b * num_channels * dimScale
                                            + 0* dimScale + h * output_width + w;
                int y_index = b * num_channels * dimScale 
                                            + 1* dimScale + h * output_width + w;
                int width_index = b * num_channels * dimScale
                                            + 2* dimScale + h * output_width + w;
                int height_index = b * num_channels * dimScale 
                                            + 3* dimScale + h * output_width + w;
                int class_index = b * num_channels * dimScale
                                            + 4* dimScale + h * output_width + w;
                float xmin = 0.f, ymin = 0.f, xmax = 0.f, ymax = 0.f;
                
                float bb_xmin = (w + 0.5 - channel_pred_data[x_index] * anchor_scale /(2*downRatio)) *downRatio;
                float bb_ymin = (h + 0.5 - channel_pred_data[y_index] * anchor_scale /(2*downRatio)) *downRatio;
                float bb_xmax = (w + 0.5 - channel_pred_data[width_index] * anchor_scale /(2*downRatio)) *downRatio;
                float bb_ymax = (h + 0.5 - channel_pred_data[height_index] * anchor_scale /(2*downRatio)) *downRatio;
                
                xmin = GET_VALID_VALUE(bb_xmin, (0.f), float(downRatio * output_width));
                ymin = GET_VALID_VALUE(bb_ymin, (0.f), float(downRatio * output_height));
                xmax = GET_VALID_VALUE(bb_xmax, (0.f), float(downRatio * output_width));
                ymax = GET_VALID_VALUE(bb_ymax, (0.f), float(downRatio * output_height));
                if((xmax - xmin) <= 0 || (ymax - ymin) <= 0)
                    continue;                                     
                
                Dtype label_score = channel_pred_data[class_index];
                
                if(label_score >= conf_thresh){
                    CenterNetInfo temp_result;
                    temp_result.set_class_id(0);
                    temp_result.set_score(label_score);
                    temp_result.set_xmin(xmin);
                    temp_result.set_xmax(xmax);
                    temp_result.set_ymin(ymin);
                    temp_result.set_ymax(ymax);
                    temp_result.set_area((xmax - xmin) * (ymax - ymin));
                    (*results)[b].push_back(temp_result);
                }
            }
        }
    }
}

template void GetCenterGridObjectResultSigmoid(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio,
                          float* channel_pred_data, const int anchor_scale, float conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results);

template void GetCenterGridObjectResultSigmoid(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio,
                          double* channel_pred_data, const int anchor_scale, double conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results);


template <typename Dtype> 
Dtype EncodeCenterGridObjectSoftMaxLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio, std::vector<int>postive_batch,
                          std::vector<Dtype> batch_sample_loss,
                          Dtype* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                          Dtype* class_label, Dtype* bottom_diff, 
                          int *count_postive, Dtype *loc_loss_value, int *match_num_gt_box, bool has_lm, Dtype* lm_loss_value){
    CHECK_EQ(num_classes, 2) << "the current version is just classfly bg_0 and face_1, two class";
    int dimScale = output_height * output_width;
    Dtype score_loss = Dtype(0.), loc_loss = Dtype(0.), lm_loss = Dtype(0.);
    if(has_lm)
        CHECK_EQ(num_channels, (14 + num_classes)) 
            << "num_channels shoule be set to including bias_x, bias_y, width, height, classes";
    else
        CHECK_EQ(num_channels, (4 + num_classes)) 
            << "num_channels shoule be set to including 4 points landmarks, &classes";

    int postive = 0;
    int gt_match_box = 0;
    SoftmaxCenterGrid(channel_pred_data, batch_size, num_classes, num_channels, output_height, output_width, has_lm);
    // 将所有值设置为 -2 的原因为，排除掉iou>0.35的一些样本，也就是说
    // 只采集那些iou<0.35的负样本.20200611,舍弃之
    #if FOCAL_LOSS_SOFTMAX
    caffe_set(batch_size * dimScale, Dtype(0.5f), class_label);
    #else
    caffe_set(batch_size * dimScale, Dtype(-1.), class_label);
    #endif
    for(int b = 0; b < batch_size; b++){
        std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > >::iterator it = all_gt_bboxes.find(b);
        if(it == all_gt_bboxes.end()){
            continue;
        }
        vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > gt_bboxes = it->second;
        int count = 0;
        for(int h = 0; h < output_height; h++){
            for(int w = 0; w < output_width; w++){
                int bg_index = b * num_channels * dimScale 
                                            + 4* dimScale + h * output_width + w;
                int face_index = b * num_channels * dimScale 
                                            + 5* dimScale + h * output_width + w;
                if(has_lm){
                    bg_index = b * num_channels * dimScale 
                                            + 14* dimScale + h * output_width + w;
                    face_index = b * num_channels * dimScale 
                                            + 15* dimScale + h * output_width + w;
                }
                Dtype class_loss = SingleSoftmaxLoss(channel_pred_data[bg_index], channel_pred_data[face_index], Dtype(-1.));
                batch_sample_loss[b * dimScale + h * output_width + w] = class_loss;
            }
        }
        std::vector<int> mask_Rf_anchor_already(dimScale, 0);
        for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
            Dtype xmin = gt_bboxes[ii].first.xmin() * output_width;
            Dtype ymin = gt_bboxes[ii].first.ymin() * output_height;
            Dtype xmax = gt_bboxes[ii].first.xmax() * output_width;
            Dtype ymax = gt_bboxes[ii].first.ymax() * output_height;
            if ((xmax - xmin) <= 0 || (ymax - ymin) <=0){
                LOG(INFO)<<"xmin: "<<xmin<<
                            ", xmax: "<<xmax<<
                            ", ymin: "<<ymin<<
                            ", ymax: "<<ymax;
                continue;
            }
            for(int h = static_cast<int>(ymin); h < static_cast<int>(ymax); h++){
                for(int w = static_cast<int>(xmin); w < static_cast<int>(xmax); w++){
                    int class_index = b * dimScale +  h * output_width + w;
                    class_label[class_index] = -10;
                }
            }
            int gt_bbox_width = static_cast<int>((xmax - xmin) * downRatio);
            int gt_bbox_height = static_cast<int>((ymax - ymin) * downRatio);
            int large_side = std::max(gt_bbox_height, gt_bbox_width);
            if(large_side >= loc_truth_scale.first && large_side < loc_truth_scale.second){
                
                for(int h = static_cast<int>(ymin); h < static_cast<int>(ymax); h++){
                    for(int w = static_cast<int>(xmin); w < static_cast<int>(xmax); w++){
                        if(mask_Rf_anchor_already[h * output_width + w] == 1) // 避免同一个anchor的中心落在多个gt里面
                            continue;
                   
                        Dtype xmin_bias = (w + 0.5 - xmin) * downRatio * 2 / anchor_scale;
                        Dtype ymin_bias = (h + 0.5 - ymin) * downRatio * 2 / anchor_scale;
                        Dtype xmax_bias = (w + 0.5 - xmax) * downRatio * 2 / anchor_scale;
                        Dtype ymax_bias = (h + 0.5 - ymax) * downRatio * 2 / anchor_scale;
                        int xmin_index = b * num_channels * dimScale
                                                    + 0* dimScale + h * output_width + w;
                        int ymin_index = b * num_channels * dimScale 
                                                    + 1* dimScale + h * output_width + w;
                        int xmax_index = b * num_channels * dimScale
                                                    + 2* dimScale + h * output_width + w;
                        int ymax_index = b * num_channels * dimScale 
                                                    + 3* dimScale + h * output_width + w;
                        int class_index = b * dimScale +  h * output_width + w;
    
                        Dtype xmin_diff, ymin_diff, xmax_diff, ymax_diff;
                        loc_loss += smoothL1_Loss(Dtype(channel_pred_data[xmin_index] - xmin_bias), &xmin_diff);
                        loc_loss += smoothL1_Loss(Dtype(channel_pred_data[ymin_index] - ymin_bias), &ymin_diff);
                        loc_loss += smoothL1_Loss(Dtype(channel_pred_data[xmax_index] - xmax_bias), &xmax_diff);
                        loc_loss += smoothL1_Loss(Dtype(channel_pred_data[ymax_index] - ymax_bias), &ymax_diff);
                        
                        bottom_diff[xmin_index] = xmin_diff;
                        bottom_diff[ymin_index] = ymin_diff;
                        bottom_diff[xmax_index] = xmax_diff;
                        bottom_diff[ymax_index] = ymax_diff;
                        if(has_lm){
                            if(gt_bboxes[ii].second.lefteye().x() > 0 && gt_bboxes[ii].second.lefteye().y() > 0 &&
                            gt_bboxes[ii].second.righteye().x() > 0 && gt_bboxes[ii].second.righteye().y() > 0 && 
                            gt_bboxes[ii].second.nose().x() > 0 && gt_bboxes[ii].second.nose().y() > 0 &&
                            gt_bboxes[ii].second.leftmouth().x() > 0 && gt_bboxes[ii].second.leftmouth().y() > 0 &&
                            gt_bboxes[ii].second.rightmouth().x() > 0 && gt_bboxes[ii].second.rightmouth().y() > 0){
                                int le_x_index = b * num_channels * dimScale + 4* dimScale + h * output_width + w;
                                int le_y_index = b * num_channels * dimScale + 5* dimScale + h * output_width + w;
                                int re_x_index = b * num_channels * dimScale + 6* dimScale + h * output_width + w;
                                int re_y_index = b * num_channels * dimScale + 7* dimScale + h * output_width + w;
                                int no_x_index = b * num_channels * dimScale + 8* dimScale + h * output_width + w;
                                int no_y_index = b * num_channels * dimScale + 9* dimScale + h * output_width + w;
                                int lm_x_index = b * num_channels * dimScale + 10* dimScale + h * output_width + w;
                                int lm_y_index = b * num_channels * dimScale + 11* dimScale + h * output_width + w;
                                int rm_x_index = b * num_channels * dimScale + 12* dimScale + h * output_width + w;
                                int rm_y_index = b * num_channels * dimScale + 13* dimScale + h * output_width + w;

                                Dtype le_x_bias = (w - gt_bboxes[ii].second.lefteye().x() * output_width) * downRatio * 2 / anchor_scale;
                                Dtype le_y_bias = (h - gt_bboxes[ii].second.lefteye().y() * output_height) * downRatio * 2 / anchor_scale;

                                Dtype re_x_bias = (w - gt_bboxes[ii].second.righteye().x() * output_width) * downRatio * 2 / anchor_scale;
                                Dtype re_y_bias = (h - gt_bboxes[ii].second.righteye().y() * output_height) * downRatio * 2 / anchor_scale;

                                Dtype no_x_bias = (w - gt_bboxes[ii].second.nose().x() * output_width) * downRatio * 2 / anchor_scale;
                                Dtype no_y_bias = (h - gt_bboxes[ii].second.nose().y() * output_height) * downRatio * 2 / anchor_scale;

                                Dtype lm_x_bias = (w - gt_bboxes[ii].second.leftmouth().x() * output_width) * downRatio * 2 / anchor_scale;
                                Dtype lm_y_bias = (h - gt_bboxes[ii].second.leftmouth().y() * output_height) * downRatio * 2 / anchor_scale;

                                Dtype rm_x_bias = (w - gt_bboxes[ii].second.rightmouth().x() * output_width) * downRatio * 2 / anchor_scale;
                                Dtype rm_y_bias = (h - gt_bboxes[ii].second.rightmouth().y() * output_height) * downRatio * 2 / anchor_scale;

                                Dtype le_x_diff, le_y_diff, re_x_diff, re_y_diff, no_x_diff, no_y_diff, lm_x_diff, lm_y_diff, rm_x_diff, rm_y_diff;

                                lm_loss += L2_Loss(Dtype(channel_pred_data[le_x_index] - le_x_bias), &le_x_diff);
                                lm_loss += L2_Loss(Dtype(channel_pred_data[le_y_index] - le_y_bias), &le_y_diff);

                                lm_loss += L2_Loss(Dtype(channel_pred_data[re_x_index] - re_x_bias), &re_x_diff);
                                lm_loss += L2_Loss(Dtype(channel_pred_data[re_y_index] - re_y_bias), &re_y_diff);

                                lm_loss += L2_Loss(Dtype(channel_pred_data[no_x_index] - no_x_bias), &no_x_diff);
                                lm_loss += L2_Loss(Dtype(channel_pred_data[no_y_index] - no_y_bias), &no_y_diff);

                                lm_loss += L2_Loss(Dtype(channel_pred_data[lm_x_index] - lm_x_bias), &lm_x_diff);
                                lm_loss += L2_Loss(Dtype(channel_pred_data[lm_y_index] - lm_y_bias), &lm_y_diff);

                                lm_loss += L2_Loss(Dtype(channel_pred_data[rm_x_index] - rm_x_bias), &rm_x_diff);
                                lm_loss += L2_Loss(Dtype(channel_pred_data[rm_y_index] - rm_y_bias), &rm_y_diff);


                                bottom_diff[le_x_index] = le_x_diff;
                                bottom_diff[le_y_index] = le_y_diff;
                                bottom_diff[re_x_index] = re_x_diff;
                                bottom_diff[re_y_index] = re_y_diff;
                                bottom_diff[no_x_index] = no_x_diff;
                                bottom_diff[no_y_index] = no_y_diff;
                                bottom_diff[lm_x_index] = lm_x_diff;
                                bottom_diff[lm_y_index] = lm_y_diff;
                                bottom_diff[rm_x_index] = rm_x_diff;
                                bottom_diff[rm_y_index] = rm_y_diff;
                            }
                        }
                        class_label[class_index] = 1;
                        mask_Rf_anchor_already[h * output_width + w] = 1;
                        count++;

                        int bg_index = b * num_channels * dimScale 
                                                    + 4* dimScale + h * output_width + w;
                        int face_index = b * num_channels * dimScale 
                                                    + 5* dimScale + h * output_width + w;
                        if(has_lm){
                            bg_index = b * num_channels * dimScale 
                                                    + 14* dimScale + h * output_width + w;
                            face_index = b * num_channels * dimScale 
                                                    + 15* dimScale + h * output_width + w;
                        }
                        Dtype class_loss = SingleSoftmaxLoss(channel_pred_data[bg_index], channel_pred_data[face_index], Dtype(1.0));
                        batch_sample_loss[b * dimScale + h * output_width + w] = class_loss;
                    }
                }
                gt_match_box ++;
            }
        }
        mask_Rf_anchor_already.clear();
        postive_batch[b] = count;
        postive += count;
    }
    #if FOCAL_LOSS_SOFTMAX
    score_loss = FocalLossSoftmax(class_label, channel_pred_data, batch_size, output_height,
                                    output_width, bottom_diff, num_channels, has_lm);
    #else
    // 计算softMax loss value 
    SelectHardSampleSoftMax(class_label, batch_sample_loss, 2, postive_batch, 
                                        output_height, output_width, num_channels, batch_size, has_lm);
    score_loss = SoftmaxWithLoss(class_label, channel_pred_data, batch_size, output_height,
                                    output_width, bottom_diff, num_channels, has_lm);
    #endif
    *count_postive = postive;
    *loc_loss_value = loc_loss;
    *lm_loss_value = lm_loss;
    *match_num_gt_box = gt_match_box;
    return score_loss;
}

template float EncodeCenterGridObjectSoftMaxLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio, std::vector<int>postive_batch,
                          std::vector<float> batch_sample_loss,
                          float* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                          float* class_label, float* bottom_diff, 
                          int *count_postive, float *loc_loss_value, int *match_num_gt_box, bool has_lm, float * lm_loss_value);

template double EncodeCenterGridObjectSoftMaxLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio, std::vector<int>postive_batch,
                          std::vector<double> batch_sample_loss,
                          double* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<std::pair<NormalizedBBox, AnnoFaceLandmarks> > > all_gt_bboxes,
                          double* class_label, double* bottom_diff, 
                          int *count_postive, double *loc_loss_value, int *match_num_gt_box, bool has_lm, double * lm_loss_value);

template <typename Dtype>
void GetCenterGridObjectResultSoftMax(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio,
                          Dtype* channel_pred_data, const int anchor_scale, Dtype conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results, bool has_lm){
    // face class 人脸类型 包括背景 + face人脸，两类
    CHECK_EQ(num_classes, 2);
    if(has_lm){
        CHECK_EQ(num_channels, 14 + num_classes);
    }else{
        CHECK_EQ(num_channels, 4 + num_classes);
    }
    
    int dimScale = output_height * output_width;
    SoftmaxCenterGrid(channel_pred_data, batch_size, num_classes, num_channels, output_height, output_width, has_lm);

    for(int b = 0; b < batch_size; b++){
        for(int h = 0; h < output_height; h++){
            for(int w = 0; w < output_width; w++){
                int x_index = b * num_channels * dimScale
                                            + 0* dimScale + h * output_width + w;
                int y_index = b * num_channels * dimScale 
                                            + 1* dimScale + h * output_width + w;
                int width_index = b * num_channels * dimScale
                                            + 2* dimScale + h * output_width + w;
                int height_index = b * num_channels * dimScale 
                                            + 3* dimScale + h * output_width + w;        
                

                float xmin = 0.f, ymin = 0.f, xmax = 0.f, ymax = 0.f;

                float bb_xmin = (w + 0.5 - channel_pred_data[x_index] * anchor_scale / (2 * downRatio)) *downRatio;
                float bb_ymin = (h + 0.5 - channel_pred_data[y_index] * anchor_scale / (2 * downRatio)) *downRatio;
                float bb_xmax = (w + 0.5 - channel_pred_data[width_index] * anchor_scale / (2 * downRatio)) *downRatio;
                float bb_ymax = (h + 0.5 - channel_pred_data[height_index] * anchor_scale / (2 * downRatio)) *downRatio;
                xmin = GET_VALID_VALUE(bb_xmin, (0.f), float(downRatio * output_width));
                ymin = GET_VALID_VALUE(bb_ymin, (0.f), float(downRatio * output_height));
                xmax = GET_VALID_VALUE(bb_xmax, (0.f), float(downRatio * output_width));
                ymax = GET_VALID_VALUE(bb_ymax, (0.f), float(downRatio * output_height));

                float le_x = 0.f, le_y = 0.f, re_x = 0.f, re_y = 0.f, no_x = 0.f, no_y = 0.f, lm_x = 0.f, lm_y = 0.f, rm_x = 0.f, rm_y = 0.f;
                if(has_lm){
                    int le_x_index = b * num_channels * dimScale + 4* dimScale + h * output_width + w;
                    int le_y_index = b * num_channels * dimScale + 5* dimScale + h * output_width + w;
                    int re_x_index = b * num_channels * dimScale + 6* dimScale + h * output_width + w;
                    int re_y_index = b * num_channels * dimScale + 7* dimScale + h * output_width + w;
                    int no_x_index = b * num_channels * dimScale + 8* dimScale + h * output_width + w;
                    int no_y_index = b * num_channels * dimScale + 9* dimScale + h * output_width + w;
                    int lm_x_index = b * num_channels * dimScale + 10* dimScale + h * output_width + w;
                    int lm_y_index = b * num_channels * dimScale + 11* dimScale + h * output_width + w;
                    int rm_x_index = b * num_channels * dimScale + 12* dimScale + h * output_width + w;
                    int rm_y_index = b * num_channels * dimScale + 13* dimScale + h * output_width + w;
                    le_x = (w - channel_pred_data[le_x_index] * anchor_scale /(2 * downRatio)) *downRatio;
                    le_y = (w - channel_pred_data[le_y_index] * anchor_scale /(2 * downRatio)) *downRatio;
                    le_x = GET_VALID_VALUE(le_x, (0.f), float(downRatio * output_width));
                    le_y = GET_VALID_VALUE(le_y, (0.f), float(downRatio * output_height));

                    re_x = (w - channel_pred_data[re_x_index] * anchor_scale /(2 * downRatio)) *downRatio;
                    re_y = (w - channel_pred_data[re_y_index] * anchor_scale /(2 * downRatio)) *downRatio;
                    re_x = GET_VALID_VALUE(re_x, (0.f), float(downRatio * output_width));
                    re_y = GET_VALID_VALUE(re_y, (0.f), float(downRatio * output_height));

                    no_x = (w - channel_pred_data[no_x_index] * anchor_scale /(2 * downRatio)) *downRatio;
                    no_y = (w - channel_pred_data[no_y_index] * anchor_scale /(2 * downRatio)) *downRatio;
                    no_x = GET_VALID_VALUE(no_x, (0.f), float(downRatio * output_width));
                    no_y = GET_VALID_VALUE(no_y, (0.f), float(downRatio * output_height));

                    lm_x = (w - channel_pred_data[lm_x_index] * anchor_scale /(2 * downRatio)) *downRatio;
                    lm_y = (w - channel_pred_data[lm_y_index] * anchor_scale /(2 * downRatio)) *downRatio;
                    lm_x = GET_VALID_VALUE(lm_x, (0.f), float(downRatio * output_width));
                    lm_y = GET_VALID_VALUE(lm_y, (0.f), float(downRatio * output_height));

                    rm_x = (w - channel_pred_data[rm_x_index] * anchor_scale /(2 * downRatio)) *downRatio;
                    rm_y = (w - channel_pred_data[rm_y_index] * anchor_scale /(2 * downRatio)) *downRatio;
                    rm_x = GET_VALID_VALUE(rm_x, (0.f), float(downRatio * output_width));
                    rm_y = GET_VALID_VALUE(rm_y, (0.f), float(downRatio * output_height));
                }

                if((xmax - xmin) <= 0 || (ymax - ymin) <= 0)
                    continue;                                     
                
                int class_index = b * num_channels * dimScale
                                            + 5* dimScale + h * output_width + w;
                if(has_lm){
                    class_index = b * num_channels * dimScale
                                            + 15* dimScale + h * output_width + w;
                }
                Dtype label_score = channel_pred_data[class_index];
                
                if(label_score >= conf_thresh){
                    CenterNetInfo temp_result;
                    temp_result.set_class_id(0);
                    temp_result.set_score(label_score);
                    temp_result.set_xmin(xmin);
                    temp_result.set_xmax(xmax);
                    temp_result.set_ymin(ymin);
                    temp_result.set_ymax(ymax);
                    temp_result.set_area((xmax - xmin) * (ymax - ymin));
                    if(has_lm){
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
                    (*results)[b].push_back(temp_result);
                }
            }
        }
    }
}

template void GetCenterGridObjectResultSoftMax(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio,
                          float* channel_pred_data, const int anchor_scale, float conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results, bool has_lm);

template void GetCenterGridObjectResultSoftMax(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const float downRatio,
                          double* channel_pred_data, const int anchor_scale, double conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results, bool has_lm);

// OHEM 样本难例挖掘， 只记录具体思路
// 1. 通过IOU 或者中心点的方式，确定正样本，也就是说，对于位置回归来说，确定回归的损失函数(loc loss), 通过样本框真实值确定样本分类
// 2. 这样会得到一张图像里面所有样本的正样本数量，以及全部的负样本数量
// 3. 正常计算正样本 + 所有负样本的loss和(位置回归损失 + 置信度损失)是一个集合(所有样本的集合)
// 针对上面的解释，假设对于160x160的featureMap，一共有25600个样本，其中有600个正样本，其余都是负样本(20000)，
// 那么，这样来计算每个样本的loss集合，
//    1).对于一个正样本，计算这个样本的loc_loss + cls_loss。
//    2).对于一个负样本，计算cls_loss(背景类的置信度得分损失)，SSD中，在计算负样本的时候
// 4. 使用NMS 去除重复的框, 针对NMS， 该如何去实现呢？ohem的作者是这样做的，
//    1).首先第一步box从哪里来，所有计算出来的预测box(160x160)
//    2).置信得分，以loss为准，具体的每个box的结构为[x1, y1, x2, y2, loss]
//    3).这样再进行nms，排除多余的框，以框IOU_Thresh > 0.7 为临界阈值
// 5. 再对剩余Loss的集合进行从大到小排列，选取一定数量的负样本
// 6. 只回归计算这样正样本和一部分负样本的总损失，以及回归相应的梯度值
template <typename Dtype>
void SelectHardSampleSoftMaxWithOHEM(){
    NOT_IMPLEMENTED;
}


// anchor匹配方式采用IOU匹配，但是还是分层的概念，每一层匹配不同大小的gt_bboxes
// 采用overlap > 0.45的为正样本，其他的均为负样本
// loss = smoothL1loss + focal loss + objectness_loss确定这一层有无目标物体
// 对于没有匹配到的某一层，则这一层loss值为0
template <typename Dtype> 
Dtype EncodeOverlapObjectSigmoidLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          Dtype* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                          Dtype* class_label, Dtype* bottom_diff, 
                          Dtype ignore_thresh, int *count_postive, Dtype *loc_loss_value){
    CHECK_EQ(num_classes, 1);
    int dimScale = output_height * output_width;
    Dtype score_loss = Dtype(0.), loc_loss = Dtype(0.);
    CHECK_EQ(num_channels, (4 + 1 + num_classes)) 
            << "num_channels shoule be set to including bias_x, bias_y, width, height, object_confidence and classes";
    for(int b = 0; b < batch_size; b++){
        int object_index = b * num_channels * dimScale + 4 * dimScale;
        for(int i = 0; i < (num_classes + 1) * dimScale; i++){
            channel_pred_data[object_index + i] = CenterSigmoid(channel_pred_data[object_index + i]);
        }
    }
    int postive = 0;
    // 采用focal loss 使用所有的负样本，作为训练
    caffe_set(batch_size * dimScale, Dtype(0.5f), class_label);
    for(int b = 0; b < batch_size; b++){
        vector<NormalizedBBox> gt_bboxes = all_gt_bboxes.find(b)->second;
        std::vector<int> mask_Rf_anchor(dimScale, 0);
        std::vector<Dtype> object_loss_temp(dimScale, Dtype(0.));
        int count = 0;
        for(int h = 0; h < output_height; h++){
            for(int w = 0; w < output_width; w++){
            int xmin_index = b * num_channels * dimScale
                                        + 0* dimScale + h * output_width + w;
            int ymin_index = b * num_channels * dimScale 
                                        + 1* dimScale + h * output_width + w;
            int xmax_index = b * num_channels * dimScale
                                        + 2* dimScale + h * output_width + w;
            int ymax_index = b * num_channels * dimScale 
                                        + 3* dimScale + h * output_width + w;
            int object_index = b * num_channels * dimScale 
                                        + 4* dimScale + h * output_width + w;
            NormalizedBBox predBox;
            Dtype center_x = (w + 0.5 - channel_pred_data[xmin_index] * anchor_scale /(2*downRatio)) / output_width;
            Dtype center_y = (h + 0.5 - channel_pred_data[ymin_index] * anchor_scale /(2*downRatio)) / output_height;
            Dtype pred_width = (std::exp(channel_pred_data[xmax_index]) * anchor_scale / downRatio) / output_width;
            Dtype pred_height = (std::exp(channel_pred_data[ymax_index]) * anchor_scale / downRatio) /output_height;
            predBox.set_xmin(center_x - pred_width / 2);
            predBox.set_xmax(center_x + pred_width / 2);
            predBox.set_ymin(center_y - pred_height / 2);
            predBox.set_ymax(center_y + pred_height / 2);
            float best_iou = 0;
            for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
                const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
                const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
                const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
                const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
                const int gt_bbox_width = static_cast<int>((xmax - xmin) * downRatio);
                const int gt_bbox_height = static_cast<int>((ymax - ymin) * downRatio);
                int large_side = std::max(gt_bbox_height, gt_bbox_width);
                if(large_side >= loc_truth_scale.first && large_side < loc_truth_scale.second){
                    float iou = YoloBBoxIou(predBox, gt_bboxes[ii]);
                    if (iou > best_iou) {
                        best_iou = iou;
                    }
                }
            }
            Dtype object_value = (channel_pred_data[object_index] - 0.);
            if(best_iou > ignore_thresh){
                object_value = 0.;
            }
            object_loss_temp[h * output_width + w] = Object_L2_Loss(object_value, &(bottom_diff[object_index]));
            }
        }
        for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
            const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
            const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
            const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
            const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
            const Dtype gt_center_x = (xmin + xmax) / 2;
            const Dtype gt_center_y = (ymin + ymax) / 2;
            const int gt_bbox_width = static_cast<int>((xmax - xmin) * downRatio);
            const int gt_bbox_height = static_cast<int>((ymax - ymin) * downRatio);
            int large_side = std::max(gt_bbox_height, gt_bbox_width);
            if(large_side >= loc_truth_scale.first && large_side < loc_truth_scale.second){
                for(int h = 0; h < output_height; h++){
                    for(int w = 0; w < output_width; w++){
                        if(mask_Rf_anchor[h * output_width + w] == 1) // 避免同一个anchor的中心落在多个gt里面
                            continue;
                        NormalizedBBox anchorBox;
                        float bb_xmin = (w - anchor_scale /(2*downRatio)) / output_width;
                        float bb_ymin = (h - anchor_scale /(2*downRatio)) / output_height;
                        float bb_xmax = (w + anchor_scale /(2*downRatio)) / output_width;
                        float bb_ymax = (h + anchor_scale /(2*downRatio)) /output_height;
                        anchorBox.set_xmin(bb_xmin);
                        anchorBox.set_xmax(bb_xmax);
                        anchorBox.set_ymin(bb_ymin);
                        anchorBox.set_ymax(bb_ymax);
                        float best_iou = YoloBBoxIou(anchorBox, gt_bboxes[ii]);
                        if(best_iou > (ignore_thresh + 0.15)){
                            int x_center_index = b * num_channels * dimScale
                                                + 0* dimScale + h * output_width + w;
                            int y_center_index = b * num_channels * dimScale 
                                                    + 1* dimScale + h * output_width + w;
                            int width_index = b * num_channels * dimScale
                                                    + 2* dimScale + h * output_width + w;
                            int height_index = b * num_channels * dimScale 
                                                    + 3* dimScale + h * output_width + w;
                            int object_index = b * num_channels * dimScale 
                                                    + 4* dimScale + h * output_width + w;
                            Dtype x_center_bias = (w + 0.5 - gt_center_x) * downRatio *2 / anchor_scale;
                            Dtype y_center_bias = (h + 0.5 - gt_center_y) * downRatio *2 / anchor_scale;
                            Dtype width_bias = std::log((xmax - xmin) * downRatio / anchor_scale);
                            Dtype height_bias = std::log((ymax - ymin) * downRatio / anchor_scale);

                            loc_loss += smoothL1_Loss(Dtype(channel_pred_data[x_center_index] - x_center_bias), &(bottom_diff[x_center_index]));
                            loc_loss += smoothL1_Loss(Dtype(channel_pred_data[y_center_index] - y_center_bias), &(bottom_diff[y_center_index]));
                            loc_loss += smoothL1_Loss(Dtype(channel_pred_data[width_index] - width_bias), &(bottom_diff[width_index]));
                            loc_loss += smoothL1_Loss(Dtype(channel_pred_data[height_index] - height_bias), &(bottom_diff[height_index]));
                            object_loss_temp[h * output_width + w] = Object_L2_Loss(Dtype(channel_pred_data[object_index] - 1.), &(bottom_diff[object_index]));
                            int class_index = b * dimScale +  h * output_width + w;
                            class_label[class_index] = 1;
                            mask_Rf_anchor[h * output_width + w] = 1;
                            count++;
                        }
                    }
                }
            }
        }
        for(unsigned ii = 0; ii < dimScale; ii++){
            loc_loss += object_loss_temp[ii];
        }
        if(count > 0){
            int gt_class_index =  b * dimScale;
            int pred_class_index = b * num_channels * dimScale + 5* dimScale;
            score_loss += FocalLossSigmoid(class_label + gt_class_index, channel_pred_data + pred_class_index, 
                                            dimScale, bottom_diff + pred_class_index);
        }else{
            score_loss += 0;
        }
        postive += count;
    }
    *count_postive = postive;
    *loc_loss_value = loc_loss;
    return score_loss;
}

template float EncodeOverlapObjectSigmoidLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          float* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                          float* class_label, float* bottom_diff, 
                          float ignore_thresh, int *count_postive, float *loc_loss_value);

template double EncodeOverlapObjectSigmoidLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          double* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                          double* class_label, double* bottom_diff, 
                          double ignore_thresh, int *count_postive, double *loc_loss_value);


template <typename Dtype>
void GetCenterOverlapResultSigmoid(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          Dtype* channel_pred_data, const int anchor_scale, Dtype conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results){
    // face class 人脸类型
    CHECK_EQ(num_classes, 1);
    CHECK_EQ(num_channels, 4 + 1 + num_classes);
    int dimScale = output_height * output_width;
    for(int b = 0; b < batch_size; b++){
        int object_index = b * num_channels * dimScale + 4 * dimScale;
        for(int i = 0; i < (num_classes + 1) * dimScale; i++){
            channel_pred_data[object_index + i] = CenterSigmoid(channel_pred_data[object_index + i]);
        }
    }

    for(int b = 0; b < batch_size; b++){
        for(int h = 0; h < output_height; h++){
            for(int w = 0; w < output_width; w++){
                int x_index = b * num_channels * dimScale
                                            + 0* dimScale + h * output_width + w;
                int y_index = b * num_channels * dimScale 
                                            + 1* dimScale + h * output_width + w;
                int width_index = b * num_channels * dimScale
                                            + 2* dimScale + h * output_width + w;
                int height_index = b * num_channels * dimScale 
                                            + 3* dimScale + h * output_width + w;
                int object_index = b * num_channels * dimScale 
                                            + 4* dimScale + h * output_width + w;
                int class_index = b * num_channels * dimScale
                                            + 5* dimScale + h * output_width + w;

                Dtype center_x = (w + 0.5 - channel_pred_data[x_index] * anchor_scale /(2*downRatio)) * downRatio;
                Dtype center_y = (h + 0.5 - channel_pred_data[y_index] * anchor_scale /(2*downRatio)) * downRatio;
                Dtype pred_width = (std::exp(channel_pred_data[width_index]) * anchor_scale / downRatio) * downRatio;
                Dtype pred_height = (std::exp(channel_pred_data[height_index]) * anchor_scale / downRatio) * downRatio;

                        
                Dtype xmin = GET_VALID_VALUE(center_x - pred_width / 2, Dtype(0.f), Dtype(downRatio * output_width));
                Dtype ymin = GET_VALID_VALUE(center_y - pred_height / 2, Dtype(0.f), Dtype(downRatio * output_height));
                Dtype xmax = GET_VALID_VALUE(center_x + pred_width / 2, Dtype(0.f), Dtype(downRatio * output_width));
                Dtype ymax = GET_VALID_VALUE(center_y + pred_height / 2, Dtype(0.f), Dtype(downRatio * output_height));

                if((xmax - xmin) <= 0 || (ymax - ymin) <= 0)
                    continue;                                     
                
                Dtype label_score = channel_pred_data[class_index] * channel_pred_data[object_index];
                
                if(label_score >= conf_thresh){
                    CenterNetInfo temp_result;
                    temp_result.set_class_id(0);
                    temp_result.set_score(label_score);
                    temp_result.set_xmin(xmin);
                    temp_result.set_xmax(xmax);
                    temp_result.set_ymin(ymin);
                    temp_result.set_ymax(ymax);
                    temp_result.set_area((xmax - xmin) * (ymax - ymin));
                    (*results)[b].push_back(temp_result);
                }
            }
        }
    }
}

template void GetCenterOverlapResultSigmoid(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          float* channel_pred_data, const int anchor_scale, float conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results);

template void GetCenterOverlapResultSigmoid(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          double* channel_pred_data, const int anchor_scale, double conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results);

}  // namespace caffe
