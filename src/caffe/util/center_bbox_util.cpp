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

#define USE_HARD_SAMPLE_SOFTMAX 1

#define USE_HARD_SAMPLE_SIGMOID 0

#define GET_VALID_VALUE(value, min, max) ((((value) >= (min) ? (value) : (min)) < (max) ? ((value) >= (min) ? (value) : (min)): (max)))

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
    float x1,x2,y1,y2, s,tx1,tx2,ty1,ty2,ts,area,weight,ov;

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
            s = input[pos].score();
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
void EncodeTruthAndPredictions(Dtype* gt_loc_data, Dtype* pred_loc_data,
                                const int output_width, const int output_height, 
                                bool share_location, const Dtype* channel_loc_data,
                                const int num_channels, std::map<int, vector<NormalizedBBox> > all_gt_bboxes){
    std::map<int, vector<NormalizedBBox> > ::iterator iter;
    int count = 0;
    CHECK_EQ(num_channels, 4);
    CHECK_EQ(share_location, true);
    int dimScale = output_height * output_width;
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
        }
    }
}
template void EncodeTruthAndPredictions(float* gt_loc_data, float* pred_loc_data,
                                const int output_width, const int output_height, 
                                bool share_location, const float* channel_loc_data,
                                const int num_channels, std::map<int, vector<NormalizedBBox> > all_gt_bboxes);
template void EncodeTruthAndPredictions(double* gt_loc_data, double* pred_loc_data,
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
    CHECK_EQ(share_location, true);
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
void get_topK(const Dtype* keep_max_data, const Dtype* loc_data, const int output_height
                  , const int output_width, const int classes, const int num_batch
                  , std::map<int, std::vector<CenterNetInfo > >* results
                  , const int loc_channels, Dtype conf_thresh, Dtype nms_thresh){
    std::vector<CenterNetInfo > batch_result;
    int dim = classes * output_width * output_height;
    int dimScale = output_width * output_height;
    CHECK_EQ(loc_channels, 4);
    for(int i = 0; i < num_batch; i++){
        std::vector<CenterNetInfo > batch_temp;
        batch_result.clear();
        for(int c = 0 ; c < classes; c++){
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
                        batch_temp.push_back(temp_result);
                    } 
                }
            }
        }
        #if 1
        hard_nms(batch_temp, &batch_result, nms_thresh);
        //soft_nms(batch_temp, &batch_result, 0.5f, nms_thresh);
        for(unsigned j = 0 ; j < batch_result.size(); j++){
            batch_result[j].set_xmin(batch_result[j].xmin() / (4 * output_width));
            batch_result[j].set_xmax(batch_result[j].xmax() / (4 * output_width));
            batch_result[j].set_ymin(batch_result[j].ymin() / (4 * output_height));
            batch_result[j].set_ymax(batch_result[j].ymax() / (4 * output_height));
        }
        #else
        for(unsigned j = 0 ; j < batch_temp.size(); j++){
            CenterNetInfo temp_result;
            temp_result.set_class_id(batch_temp[j].class_id());
            temp_result.set_score(batch_temp[j].score());
            temp_result.set_xmin(batch_temp[j].xmin() / (4 * output_width));
            temp_result.set_xmax(batch_temp[j].xmax() / (4 * output_width));
            temp_result.set_ymin(batch_temp[j].ymin() / (4 * output_height));
            temp_result.set_ymax(batch_temp[j].ymax() / (4 * output_height));
            batch_result.push_back(temp_result);
        }
        #endif
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

// 置信度得分,用逻辑回归来做,loss_delta梯度值,既前向又后向
template <typename Dtype>
void EncodeYoloObject(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int net_width, const int net_height,
                          Dtype* channel_pred_data,
                          std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
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
        vector<NormalizedBBox> gt_bboxes = all_gt_bboxes.find(b)->second;
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
                        float iou = YoloBBoxIou(predBox, gt_bboxes[ii]);
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
            const Dtype xmin = gt_bboxes[ii].xmin();
            const Dtype ymin = gt_bboxes[ii].ymin();
            const Dtype xmax = gt_bboxes[ii].xmax();
            const Dtype ymax = gt_bboxes[ii].ymax();
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
                int class_lable = gt_bboxes[ii].label() - 1; 
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
                float iou = YoloBBoxIou(predBox, gt_bboxes[ii]);
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
                              std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                              std::vector<int> mask_bias, std::vector<std::pair<float, float> >bias_scale, 
                              float* bottom_diff, float ignore_thresh, YoloScoreShow *Score);
template void EncodeYoloObject(const int batch_size, const int num_channels, const int num_classes,
                              const int output_width, const int output_height, 
                              const int net_width, const int net_height,
                              double* channel_pred_data,
                              std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                              std::vector<int> mask_bias, std::vector<std::pair<double, double> >bias_scale, 
                              double* bottom_diff, double ignore_thresh, YoloScoreShow *Score);



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


// 每一层使用感受野作为anchor,此anchor只匹配相对应大小范围的gt_boxes, 
// anchor 生成的方式是按照感受野的大小来生成的,所以每层只有一个感受野大小的anchor, 用于匹配相应的gt_boxes;
// anchor 的匹配方式是按照每个anchor中心落在真实框之内匹配呢？,还是直接基于每个格子中心来进行预测呢？最终直接使用中心来
// 预测,因为我需要使用objectness, 去预测这个位置是否有框。不过好像也可以,对于有物体的框的位置,直接设置为有物体
// loss -01: loc loss 
// loss -02: objectness loss是否有物体的概率,每个方格是否有物体, 1-有物体,0-没有物体
// loss -03: class loss 分类概率, 使用focalloss,对所有负样本进行计算
// 只针对单类物体,二分类物体检测
template <typename Dtype> 
Dtype EncodeCenterGridObjectSigmoidLoss(const int batch_size, const int num_channels, const int num_classes,
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
    #if USE_HARD_SAMPLE_SIGMOID
    caffe_set(batch_size * dimScale, Dtype(-1.), class_label);
    #else
    caffe_set(batch_size * dimScale, Dtype(0.5f), class_label);
    #endif
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
                float bb_xmin = (w - channel_pred_data[xmin_index] * anchor_scale /(2*downRatio)) / output_width;
                float bb_ymin = (h - channel_pred_data[ymin_index] * anchor_scale /(2*downRatio)) / output_height;
                float bb_xmax = (w - channel_pred_data[xmax_index] * anchor_scale /(2*downRatio)) / output_width;
                float bb_ymax = (h - channel_pred_data[ymax_index] * anchor_scale /(2*downRatio)) /output_height;
                predBox.set_xmin(bb_xmin);
                predBox.set_xmax(bb_xmax);
                predBox.set_ymin(bb_ymin);
                predBox.set_ymax(bb_ymax);
                float best_iou = 0;
                for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
                    float iou = YoloBBoxIou(predBox, gt_bboxes[ii]);
                    if (iou > best_iou) {
                        best_iou = iou;
                    }
                }
                Dtype object_value = (channel_pred_data[object_index] - 0.), object_diff;
                if(best_iou > ignore_thresh){
                    object_value = 0.;
                }
                object_loss_temp[h * output_width + w] = Object_L2_Loss(object_value, &object_diff);
                bottom_diff[object_index] = object_diff;
            }
        }
        for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
            const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
            const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
            const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
            const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
            const int gt_bbox_width = static_cast<int>((xmax - xmin) * downRatio);
            const int gt_bbox_height = static_cast<int>((ymax - ymin) * downRatio);
            int large_side = std::max(gt_bbox_height, gt_bbox_width);
            if(large_side >= loc_truth_scale.first && large_side < loc_truth_scale.second){
                for(int h = static_cast<int>(ymin); h < static_cast<int>(ymax); h++){
                    for(int w = static_cast<int>(xmin); w < static_cast<int>(xmax); w++){
                        if(w + (anchor_scale/downRatio) / 2 >= output_width - 1)
                            continue;
                        if(h + (anchor_scale/downRatio) / 2 >= output_height - 1)
                            continue;
                        if(w - (anchor_scale/downRatio) / 2 < 0)
                            continue;
                        if(h - (anchor_scale/downRatio) / 2 < 0)
                            continue;
                        if(mask_Rf_anchor[h * output_width + w] == 1) // 避免同一个anchor的中心落在多个gt里面
                            continue;
                        Dtype xmin_bias = (w - xmin) * downRatio *2 / anchor_scale;
                        Dtype ymin_bias = (h - ymin) * downRatio *2 / anchor_scale;
                        Dtype xmax_bias = (w - xmax) * downRatio *2 / anchor_scale;
                        Dtype ymax_bias = (h - ymax) * downRatio *2 / anchor_scale;
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
                        Dtype xmin_diff, ymin_diff, xmax_diff, ymax_diff, object_diff;
                        loc_loss += L2_Loss(Dtype(channel_pred_data[xmin_index] - xmin_bias), &xmin_diff);
                        loc_loss += L2_Loss(Dtype(channel_pred_data[ymin_index] - ymin_bias), &ymin_diff);
                        loc_loss += L2_Loss(Dtype(channel_pred_data[xmax_index] - xmax_bias), &xmax_diff);
                        loc_loss += L2_Loss(Dtype(channel_pred_data[ymax_index] - ymax_bias), &ymax_diff);
                        object_loss_temp[h * output_width + w] = Object_L2_Loss(Dtype(channel_pred_data[object_index] - 1.), 
                                                                    &object_diff);

                        bottom_diff[xmin_index] = xmin_diff;
                        bottom_diff[ymin_index] = ymin_diff;
                        bottom_diff[xmax_index] = xmax_diff;
                        bottom_diff[ymax_index] = ymax_diff;
                        bottom_diff[object_index] = object_diff;
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
        for(unsigned ii = 0; ii < dimScale; ii++){
            loc_loss += object_loss_temp[ii];
        }
        if(count > 0){
            int gt_class_index =  b * dimScale;
            int pred_class_index = b * num_channels * dimScale + 5* dimScale;
            #if USE_HARD_SAMPLE_SIGMOID
            SelectHardSampleSigmoid(class_label + gt_class_index, channel_pred_data + pred_class_index, 
                                    3, count, output_height, output_width, num_channels);
            #endif
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

template float EncodeCenterGridObjectSigmoidLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          float* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                          float* class_label, float* bottom_diff, 
                          float ignore_thresh, int *count_postive, float *loc_loss_value);

template double EncodeCenterGridObjectSigmoidLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          double* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                          double* class_label, double* bottom_diff, 
                          double ignore_thresh, int *count_postive, double *loc_loss_value);

template <typename Dtype>
void GetCenterGridObjectResultSigmoid(const int batch_size, const int num_channels, const int num_classes,
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

                float bb_xmin = (w - channel_pred_data[x_index] * anchor_scale /(2*downRatio)) *downRatio;
                float bb_ymin = (h - channel_pred_data[y_index] * anchor_scale /(2*downRatio)) *downRatio;
                float bb_xmax = (w - channel_pred_data[width_index] * anchor_scale /(2*downRatio)) *downRatio;
                float bb_ymax = (h - channel_pred_data[height_index] * anchor_scale /(2*downRatio)) *downRatio;
                
                float xmin = GET_VALID_VALUE(bb_xmin, (0.f), float(downRatio * output_width));
                float ymin = GET_VALID_VALUE(bb_ymin, (0.f), float(downRatio * output_height));
                float xmax = GET_VALID_VALUE(bb_xmax, (0.f), float(downRatio * output_width));
                float ymax = GET_VALID_VALUE(bb_ymax, (0.f), float(downRatio * output_height));

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

template void GetCenterGridObjectResultSigmoid(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          float* channel_pred_data, const int anchor_scale, float conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results);

template void GetCenterGridObjectResultSigmoid(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          double* channel_pred_data, const int anchor_scale, double conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results);


template <typename Dtype> 
Dtype EncodeCenterGridObjectSoftMaxLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio, std::vector<int>postive_batch,
                          std::vector<Dtype> batch_sample_loss, std::vector<int> mask_Rf_anchor,
                          Dtype* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                          Dtype* class_label, Dtype* bottom_diff, 
                          int *count_postive, Dtype *loc_loss_value, int *match_num_gt_box){
    CHECK_EQ(num_classes, 2) << "the current version is just classfly bg_0 and face_1, two class";
    int dimScale = output_height * output_width;
    Dtype score_loss = Dtype(0.), loc_loss = Dtype(0.);
    CHECK_EQ(num_channels, (4 + num_classes)) << "num_channels shoule be set to including bias_x, bias_y, width, height, classes";

    int postive = 0;
    int gt_match_box = 0;
    // 将所有值设置为 -2 的原因为，排除掉iou>0.35的一些样本，也就是说
    // 只采集那些iou<0.35的负样本
    caffe_set(batch_size * dimScale, Dtype(-2.), class_label); 
    for(int b = 0; b < batch_size; b++){
        vector<NormalizedBBox> gt_bboxes = all_gt_bboxes.find(b)->second;
        int count = 0;
        for(int h = 0; h < output_height; h++){
            for(int w = 0; w < output_width; w++){
                int bg_index = b * num_channels * dimScale 
                                            + 4* dimScale + h * output_width + w;
                int face_index = b * num_channels * dimScale 
                                            + 5* dimScale + h * output_width + w;
                Dtype class_loss = SingleSoftmaxLoss(channel_pred_data[bg_index], channel_pred_data[face_index], Dtype(-1.));
                batch_sample_loss[b * dimScale + h * output_width + w] = class_loss;
                #if 1
                int class_index = b * dimScale +  h * output_width + w;
                for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
                    const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
                    const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
                    const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
                    const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
                    const int gt_bbox_width = static_cast<int>((xmax - xmin) * downRatio);
                    const int gt_bbox_height = static_cast<int>((ymax - ymin) * downRatio);
                    int large_side = std::max(gt_bbox_height, gt_bbox_width);
                    if(large_side >= loc_truth_scale.first && large_side < loc_truth_scale.second){
                        NormalizedBBox anchor_bbox;
                        float an_xmin = GET_VALID_VALUE((float)(w - float(anchor_scale/downRatio) / 2) / output_width, 0.f, 1.f);
                        float an_ymin = GET_VALID_VALUE((float)(h - float(anchor_scale/downRatio) / 2) / output_height, 0.f, 1.f);
                        float an_xmax = GET_VALID_VALUE((float)(w + float(anchor_scale/downRatio) / 2) / output_width, 0.f, 1.f);
                        float an_ymax = GET_VALID_VALUE((float)(h + float(anchor_scale/downRatio) / 2) / output_height, 0.f, 1.f);
                        anchor_bbox.set_xmin(an_xmin);
                        anchor_bbox.set_xmax(an_xmax);
                        anchor_bbox.set_ymin(an_ymin);
                        anchor_bbox.set_ymax(an_ymax);
                        if(YoloBBoxIou(anchor_bbox, gt_bboxes[ii]) < 0.35){
                            class_label[class_index] = -1;
                        }
                    }
                }
                #endif
            }
        }
        for(unsigned ii = 0; ii < gt_bboxes.size(); ii++){
            const Dtype xmin = gt_bboxes[ii].xmin() * output_width;
            const Dtype ymin = gt_bboxes[ii].ymin() * output_height;
            const Dtype xmax = gt_bboxes[ii].xmax() * output_width;
            const Dtype ymax = gt_bboxes[ii].ymax() * output_height;
            const int gt_bbox_width = static_cast<int>((xmax - xmin) * downRatio);
            const int gt_bbox_height = static_cast<int>((ymax - ymin) * downRatio);
            int large_side = std::max(gt_bbox_height, gt_bbox_width);
            if(large_side >= loc_truth_scale.first && large_side < loc_truth_scale.second){
                for(int h = static_cast<int>(ymin); h < static_cast<int>(ymax); h++){
                    for(int w = static_cast<int>(xmin); w < static_cast<int>(xmax); w++){
                        if(w + (anchor_scale/downRatio) / 2 >= output_width - 1)
                            continue;
                        if(h + (anchor_scale/downRatio) / 2 >= output_height - 1)
                            continue;
                        if(w - (anchor_scale/downRatio) / 2 < 0)
                            continue;
                        if(h - (anchor_scale/downRatio) / 2 < 0)
                            continue;
                        if(mask_Rf_anchor[h * output_width + w] == 1) // 避免同一个anchor的中心落在多个gt里面
                            continue;

                        #if 0
                        NormalizedBBox anchor_bbox;
                        float an_xmin = GET_VALID_VALUE((float)(w - float(anchor_scale/downRatio) / 2) / output_width, 0.f, 1.f);
                        float an_ymin = GET_VALID_VALUE((float)(h - float(anchor_scale/downRatio) / 2) / output_height, 0.f, 1.f);
                        float an_xmax = GET_VALID_VALUE((float)(w + float(anchor_scale/downRatio) / 2) / output_width, 0.f, 1.f);
                        float an_ymax = GET_VALID_VALUE((float)(h + float(anchor_scale/downRatio) / 2) / output_height, 0.f, 1.f);
                        anchor_bbox.set_xmin(an_xmin);
                        anchor_bbox.set_xmax(an_xmax);
                        anchor_bbox.set_ymin(an_ymin);
                        anchor_bbox.set_ymax(an_ymax);
                        if(BBoxCoverage(gt_bboxes[ii], anchor_bbox) < 0.5){
                            continue;
                        }
                        #endif
                        
                        Dtype xmin_bias = (w - xmin) * downRatio * 2 / anchor_scale;
                        Dtype ymin_bias = (h - ymin) * downRatio * 2 / anchor_scale;
                        Dtype xmax_bias = (w - xmax) * downRatio * 2 / anchor_scale;
                        Dtype ymax_bias = (h - ymax) * downRatio * 2 / anchor_scale;
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
                        Dtype xmin_loss, ymin_loss, xmax_loss, ymax_loss, single_total_loss;
                        xmin_loss = smoothL1_Loss(Dtype(channel_pred_data[xmin_index] - xmin_bias), &xmin_diff);
                        ymin_loss = smoothL1_Loss(Dtype(channel_pred_data[ymin_index] - ymin_bias), &ymin_diff);
                        xmax_loss = smoothL1_Loss(Dtype(channel_pred_data[xmax_index] - xmax_bias), &xmax_diff);
                        ymax_loss = smoothL1_Loss(Dtype(channel_pred_data[ymax_index] - ymax_bias), &ymax_diff);
                        single_total_loss = xmin_loss + ymin_loss + xmax_loss + ymax_loss;
                        loc_loss += single_total_loss;
                        bottom_diff[xmin_index] = xmin_diff;
                        bottom_diff[ymin_index] = ymin_diff;
                        bottom_diff[xmax_index] = xmax_diff;
                        bottom_diff[ymax_index] = ymax_diff;
                        class_label[class_index] = 1;
                        mask_Rf_anchor[h * output_width + w] = 1;
                        count++;
                        int bg_index = b * num_channels * dimScale 
                                                    + 4* dimScale + h * output_width + w;
                        int face_index = b * num_channels * dimScale 
                                                    + 5* dimScale + h * output_width + w;
                        Dtype class_loss = SingleSoftmaxLoss(channel_pred_data[bg_index], channel_pred_data[face_index], Dtype(1.0));
                        batch_sample_loss[b * dimScale + h * output_width + w] = single_total_loss + class_loss;
                    }
                }
                gt_match_box ++;
            }
        }
        postive_batch[b] = count;
        postive += count;
    }
    // 计算softMax loss value 
    SelectHardSampleSoftMax(class_label, batch_sample_loss, 3, postive_batch, 
                                        output_height, output_width, num_channels, batch_size);
    score_loss = SoftmaxLossEntropy(class_label, channel_pred_data, batch_size, output_height,
                                    output_width, bottom_diff, num_channels);
    *count_postive = postive;
    *loc_loss_value = loc_loss;
    *match_num_gt_box = gt_match_box;
    return score_loss;
}

template float EncodeCenterGridObjectSoftMaxLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio, std::vector<int>postive_batch,
                          std::vector<float> batch_sample_loss, std::vector<int> mask_Rf_anchor,
                          float* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                          float* class_label, float* bottom_diff, 
                          int *count_postive, float *loc_loss_value, int *match_num_gt_box);

template double EncodeCenterGridObjectSoftMaxLoss(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio, std::vector<int>postive_batch,
                          std::vector<double> batch_sample_loss, std::vector<int> mask_Rf_anchor,
                          double* channel_pred_data, const int anchor_scale, 
                          std::pair<int, int> loc_truth_scale,
                          std::map<int, vector<NormalizedBBox> > all_gt_bboxes,
                          double* class_label, double* bottom_diff, 
                          int *count_postive, double *loc_loss_value, int *match_num_gt_box);

template <typename Dtype>
void GetCenterGridObjectResultSoftMax(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          Dtype* channel_pred_data, const int anchor_scale, Dtype conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results){
    // face class 人脸类型 包括背景 + face人脸，两类
    CHECK_EQ(num_classes, 2);
    CHECK_EQ(num_channels, 4 + num_classes);
    int dimScale = output_height * output_width;
    SoftmaxCenterGrid(channel_pred_data, batch_size, num_classes, num_channels, output_height, output_width);

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
                                            + 5* dimScale + h * output_width + w;

                float bb_xmin = (w - channel_pred_data[x_index] * anchor_scale /(2 * downRatio)) *downRatio;
                float bb_ymin = (h - channel_pred_data[y_index] * anchor_scale /(2 * downRatio)) *downRatio;
                float bb_xmax = (w - channel_pred_data[width_index] * anchor_scale /(2 * downRatio)) *downRatio;
                float bb_ymax = (h - channel_pred_data[height_index] * anchor_scale /(2 * downRatio)) *downRatio;
                
                float xmin = GET_VALID_VALUE(bb_xmin, (0.f), float(downRatio * output_width));
                float ymin = GET_VALID_VALUE(bb_ymin, (0.f), float(downRatio * output_height));
                float xmax = GET_VALID_VALUE(bb_xmax, (0.f), float(downRatio * output_width));
                float ymax = GET_VALID_VALUE(bb_ymax, (0.f), float(downRatio * output_height));

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

template void GetCenterGridObjectResultSoftMax(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          float* channel_pred_data, const int anchor_scale, float conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results);

template void GetCenterGridObjectResultSoftMax(const int batch_size, const int num_channels, const int num_classes,
                          const int output_width, const int output_height, 
                          const int downRatio,
                          double* channel_pred_data, const int anchor_scale, double conf_thresh, 
                          std::map<int, std::vector<CenterNetInfo > >* results);

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
