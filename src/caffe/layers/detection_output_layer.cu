#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->gpu_data();
  const Dtype* prior_data = bottom[2]->gpu_data();
  const int num = bottom[0]->num();

  // Decode predictions.
  Dtype* bbox_data = bbox_preds_.mutable_gpu_data();
  const int loc_count = bbox_preds_.count();
  const bool clip_bbox = false;
  DecodeBBoxesGPU<Dtype>(loc_count, loc_data, prior_data, code_type_,
      variance_encoded_in_target_, num_priors_, share_location_,
      num_loc_classes_, background_label_id_, clip_bbox, bbox_data);
  // Retrieve all decoded location predictions.
  const Dtype* bbox_cpu_data;
  if (!share_location_) {
    Dtype* bbox_permute_data = bbox_permute_.mutable_gpu_data();
    PermuteDataGPU<Dtype>(loc_count, bbox_data, num_loc_classes_, num_priors_,
        4, bbox_permute_data);
    bbox_cpu_data = bbox_permute_.cpu_data();
  } else {
    bbox_cpu_data = bbox_preds_.cpu_data();
  }
  // Retrieve all confidences.
  Dtype* conf_permute_data = conf_permute_.mutable_gpu_data();
  PermuteDataGPU<Dtype>(bottom[1]->count(), bottom[1]->gpu_data(),
      num_classes_, num_priors_, 1, conf_permute_data);
  const Dtype* conf_cpu_data = conf_permute_.cpu_data();

  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> > indices;
    int num_det = 0;
    const int conf_idx = i * num_classes_ * num_priors_;
    int bbox_idx;
    if (share_location_) {
      bbox_idx = i * num_priors_ * 4;
    } else {
      bbox_idx = conf_idx * 4;
    }
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      const Dtype* cur_conf_data = conf_cpu_data + conf_idx + c * num_priors_;
      const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        cur_bbox_data += c * num_priors_ * 4;
      }
      ApplyNMSFast(cur_bbox_data, cur_conf_data, num_priors_,
          confidence_threshold_, nms_threshold_, eta_, top_k_, &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          float score = conf_cpu_data[conf_idx + label * num_priors_ + idx];
          score_index_pairs.push_back(std::make_pair(
                  score, std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }

  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  if(attri_type_ == DetectionOutputParameter_AnnoataionAttriType_FACE){
    // Retrieve all confidences.
    Dtype* blur_permute_data = blur_permute_.mutable_gpu_data();
    PermuteDataGPU<Dtype>(bottom[3]->count(), bottom[3]->gpu_data(),
        num_blur_, num_priors_, 1, blur_permute_data);
    const Dtype* blur_cpu_data = blur_permute_.cpu_data();

    // Retrieve all confidences.
    Dtype* occlu_permute_data = occlu_permute_.mutable_gpu_data();
    PermuteDataGPU<Dtype>(bottom[4]->count(), bottom[4]->gpu_data(),
        num_occlusion_, num_priors_, 1, occlu_permute_data);
    const Dtype* occlu_cpu_data = occlu_permute_.cpu_data();

    top_shape.push_back(9);
    Dtype* top_data;
    if (num_kept == 0) {
      LOG(INFO) << "Couldn't find any detections";
      top_shape[2] = num;
      top[0]->Reshape(top_shape);
      top_data = top[0]->mutable_cpu_data();
      caffe_set<Dtype>(top[0]->count(), -1, top_data);
      // Generate fake results per image.
      for (int i = 0; i < num; ++i) {
        top_data[0] = i;
        top_data += 9;
      }
    } else {
      top[0]->Reshape(top_shape);
      top_data = top[0]->mutable_cpu_data();
    }
    int count = 0;
    boost::filesystem::path output_directory(output_directory_);
    for (int i = 0; i < num; ++i) {
      const int conf_idx = i * num_classes_ * num_priors_;
      const int blur_idx = i * num_blur_ * num_priors_;
      const int occlu_idx = i * num_occlusion_ * num_priors_;
      int bbox_idx;
      if (share_location_) {
        bbox_idx = i * num_priors_ * 4;
      } else {
        bbox_idx = conf_idx * 4;
      }
      for (map<int, vector<int> >::iterator it = all_indices[i].begin();
          it != all_indices[i].end(); ++it) {
        int label = it->first;
        vector<int>& indices = it->second;
        if (need_save_) {
          CHECK(label_to_name_.find(label) != label_to_name_.end())
            << "Cannot find label: " << label << " in the label map.";
          CHECK_LT(name_count_, names_.size());
        }
        const Dtype* cur_conf_data =
          conf_cpu_data + conf_idx + label * num_priors_;
        const Dtype* cur_blur_data =
          blur_cpu_data + blur_idx;
        const Dtype* cur_occlu_data =
          occlu_cpu_data + occlu_idx;
        const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
        if (!share_location_) {
          cur_bbox_data += label * num_priors_ * 4;
        }
        for (int j = 0; j < indices.size(); ++j) {
          int idx = indices[j];
          top_data[count * 9] = i;
          top_data[count * 9 + 1] = label;
          top_data[count * 9 + 2] = cur_conf_data[idx];
          for (int k = 0; k < 4; ++k) {
            top_data[count * 9 + 3 + k] = cur_bbox_data[idx * 4 + k];
          }
          int blur_index = 0; int occlu_index = 0;
          float blur_temp =0; float occlu_temp =0.0;
          for (int ii = 0; ii< 3; ii++ )
          {
            if (blur_temp <  cur_blur_data[idx+ii])
            {
              blur_index = ii;
              blur_temp = cur_blur_data[idx+ii];
            }
            if (occlu_temp <  cur_occlu_data[idx+ii])
            {
              occlu_index = ii;
              occlu_temp = cur_occlu_data[idx+ii];
            }
          }
          top_data[count * 9 + 7] = blur_index;
          top_data[count * 9 + 8] = occlu_index;
          ++count;
        }
      }
    }
  }else if(attri_type_ == DetectionOutputParameter_AnnoataionAttriType_LPnumber){
    // Retrieve all chineselp.
    Dtype* chi_permute_data = chinese_permute_.mutable_gpu_data();
    PermuteDataGPU<Dtype>(bottom[3]->count(), bottom[3]->gpu_data(),
        num_chinese_, num_priors_, 1, chi_permute_data);
    const Dtype* chi_cpu_data = chinese_permute_.cpu_data();

    // Retrieve all englishlp.
    Dtype* eng_permute_data = english_permute_.mutable_gpu_data();
    PermuteDataGPU<Dtype>(bottom[4]->count(), bottom[4]->gpu_data(),
        num_english_, num_priors_, 1, eng_permute_data);
    const Dtype* eng_cpu_data = english_permute_.cpu_data();

    // Retrieve all letterlp.
    Dtype* letter_1_permute_data = letter_1_permute_.mutable_gpu_data();
    PermuteDataGPU<Dtype>(bottom[5]->count(), bottom[5]->gpu_data(),
        num_letter_, num_priors_, 1, letter_1_permute_data);
    const Dtype* lettet_1_cpu_data = letter_1_permute_.cpu_data();

    // Retrieve all letterlp.
    Dtype* letter_2_permute_data = letter_2_permute_.mutable_gpu_data();
    PermuteDataGPU<Dtype>(bottom[6]->count(), bottom[6]->gpu_data(),
        num_letter_, num_priors_, 1, letter_2_permute_data);
    const Dtype* lettet_2_cpu_data = letter_2_permute_.cpu_data();

    // Retrieve all letterlp.
    Dtype* letter_3_permute_data = letter_3_permute_.mutable_gpu_data();
    PermuteDataGPU<Dtype>(bottom[7]->count(), bottom[7]->gpu_data(),
        num_letter_, num_priors_, 1, letter_3_permute_data);
    const Dtype* lettet_3_cpu_data = letter_3_permute_.cpu_data();

    // Retrieve all letterlp.
    Dtype* letter_4_permute_data = letter_4_permute_.mutable_gpu_data();
    PermuteDataGPU<Dtype>(bottom[8]->count(), bottom[8]->gpu_data(),
        num_letter_, num_priors_, 1, letter_4_permute_data);
    const Dtype* lettet_4_cpu_data = letter_4_permute_.cpu_data();

    // Retrieve all letterlp.
    Dtype* letter_5_permute_data = letter_5_permute_.mutable_gpu_data();
    PermuteDataGPU<Dtype>(bottom[9]->count(), bottom[9]->gpu_data(),
        num_letter_, num_priors_, 1, letter_5_permute_data);
    const Dtype* lettet_5_cpu_data = letter_5_permute_.cpu_data();

    top_shape.push_back(14);
    Dtype* top_data;
    if (num_kept == 0) {
      LOG(INFO) << "Couldn't find any detections";
      top_shape[2] = num;
      top[0]->Reshape(top_shape);
      top_data = top[0]->mutable_cpu_data();
      caffe_set<Dtype>(top[0]->count(), -1, top_data);
      // Generate fake results per image.
      for (int i = 0; i < num; ++i) {
        top_data[0] = i;
        top_data += 14;
      }
    } else {
      top[0]->Reshape(top_shape);
      top_data = top[0]->mutable_cpu_data();
    }
    int count = 0;
    boost::filesystem::path output_directory(output_directory_);
    for (int i = 0; i < num; ++i) {
      const int conf_idx = i * num_classes_ * num_priors_;
      const int chi_idx = i * num_chinese_ * num_priors_;
      const int eng_idx = i * num_english_ * num_priors_;
      const int let_idx = i * num_letter_ * num_priors_;
      int bbox_idx;
      if (share_location_) {
        bbox_idx = i * num_priors_ * 4;
      } else {
        bbox_idx = conf_idx * 4;
      }
      for (map<int, vector<int> >::iterator it = all_indices[i].begin();
          it != all_indices[i].end(); ++it) {
        int label = it->first;
        vector<int>& indices = it->second;
        if (need_save_) {
          CHECK(label_to_name_.find(label) != label_to_name_.end())
            << "Cannot find label: " << label << " in the label map.";
          CHECK_LT(name_count_, names_.size());
        }
        const Dtype* cur_conf_data =
          conf_cpu_data + conf_idx + label * num_priors_;
        const Dtype* cur_chi_data = chi_cpu_data + chi_idx;
        const Dtype* cur_eng_data = eng_cpu_data + eng_idx;
        const Dtype* cur_let1_data = lettet_1_cpu_data + let_idx;
        const Dtype* cur_let2_data = lettet_2_cpu_data + let_idx;
        const Dtype* cur_let3_data = lettet_3_cpu_data + let_idx;
        const Dtype* cur_let4_data = lettet_4_cpu_data + let_idx;
        const Dtype* cur_let5_data = lettet_5_cpu_data + let_idx;

        const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
        if (!share_location_) {
          cur_bbox_data += label * num_priors_ * 4;
        }
        for (int j = 0; j < indices.size(); ++j) {
          int idx = indices[j];
          top_data[count * 14] = i;
          top_data[count * 14 + 1] = label;
          top_data[count * 14 + 2] = cur_conf_data[idx];
          for (int k = 0; k < 4; ++k) {
            top_data[count * 14 + 3 + k] = cur_bbox_data[idx * 4 + k];
          }
          int chi_index = 0; int eng_index = 0; int let1_index = 0; int let2_index = 0;
          int let3_index = 0; int let4_index = 0;int let5_index = 0;
          float chi_temp =0; float eng_temp =0.0; float let1_temp =0; float let2_temp =0.0;
          float let3_temp =0; float let4_temp =0.0; float let5_temp =0;
          for (int ii = 0; ii< num_chinese_; ii++ )
          {
            if (chi_temp <  cur_chi_data[idx+ii])
            {
              chi_index = ii;
              chi_temp = cur_chi_data[idx+ii];
            }
          }
          for (int ii = 0; ii< num_english_; ii++ ){
            if (eng_temp <  cur_eng_data[idx+ii])
            {
              eng_index = ii;
              eng_temp = cur_eng_data[idx+ii];
            }
          }
          for (int ii = 0; ii< num_letter_; ii++ ){
            if (let1_temp <  cur_let1_data[idx+ii])
            {
              let1_index = ii;
              let1_temp = cur_let1_data[idx+ii];
            }
          }
          for (int ii = 0; ii< num_letter_; ii++ ){
            if (let2_temp <  cur_let2_data[idx+ii])
            {
              let2_index = ii;
              let2_temp = cur_let2_data[idx+ii];
            }
          }
          for (int ii = 0; ii< num_letter_; ii++ ){
            if (let3_temp <  cur_let3_data[idx+ii])
            {
              let3_index = ii;
              let3_temp = cur_let3_data[idx+ii];
            }
          }
          for (int ii = 0; ii< num_letter_; ii++ ){
            if (let4_temp <  cur_let4_data[idx+ii])
            {
              let4_index = ii;
              let4_temp = cur_let4_data[idx+ii];
            }
          }
          for (int ii = 0; ii< num_letter_; ii++ ){
            if (let5_temp <  cur_let5_data[idx+ii])
            {
              let5_index = ii;
              let5_temp = cur_let5_data[idx+ii];
            }
          }

          top_data[count * 14 + 7] = chi_index;
          top_data[count * 14 + 8] = eng_index;
          top_data[count * 14 + 9] = let1_index;
          top_data[count * 14 + 10] = let2_index;
          top_data[count * 14 + 11] = let3_index;
          top_data[count * 14 + 12] = let4_index;
          top_data[count * 14 + 13] = let5_index;
          ++count;
        }
      }
    }
  }
  if (visualize_) {
#ifdef USE_OPENCV
    vector<cv::Mat> cv_imgs;
    this->data_transformer_->TransformInv(bottom[3], &cv_imgs);
    vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
    VisualizeBBox(cv_imgs, top[0], visualize_threshold_, colors,
        label_to_display_name_, save_file_);
#endif  // USE_OPENCV
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DetectionOutputLayer);

}  // namespace caffe
