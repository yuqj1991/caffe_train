#include <algorithm>
#include <vector>

#include "caffe/util/bbox_util.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

void GroupObjectBBoxes(const AnnotatedDatum& anno_datum,
                       vector<NormalizedBBox>* object_bboxes) {
  object_bboxes->clear();
  for (int i = 0; i < anno_datum.annotation_group_size(); ++i) {
    const AnnotationGroup& anno_group = anno_datum.annotation_group(i);
    for (int j = 0; j < anno_group.annotation_size(); ++j) {
      const Annotation& anno = anno_group.annotation(j);
      object_bboxes->push_back(anno.bbox());
    }
  }
}

bool SatisfySampleConstraint(const NormalizedBBox& sampled_bbox,
                             const vector<NormalizedBBox>& object_bboxes,
                             const SampleConstraint& sample_constraint) {
  bool has_jaccard_overlap = sample_constraint.has_min_jaccard_overlap() ||
      sample_constraint.has_max_jaccard_overlap();
  bool has_sample_coverage = sample_constraint.has_min_sample_coverage() ||
      sample_constraint.has_max_sample_coverage();
  bool has_object_coverage = sample_constraint.has_min_object_coverage() ||
      sample_constraint.has_max_object_coverage();
  bool satisfy = !has_jaccard_overlap && !has_sample_coverage &&
      !has_object_coverage;
  if (satisfy) {
    // By default, the sampled_bbox is "positive" if no constraints are defined.
    return true;
  }
  // Check constraints.
  bool found = false;
  for (int i = 0; i < object_bboxes.size(); ++i) {
    const NormalizedBBox& object_bbox = object_bboxes[i];
    // Test jaccard overlap.
    if (has_jaccard_overlap) {
      const float jaccard_overlap = JaccardOverlap(sampled_bbox, object_bbox);
      if (sample_constraint.has_min_jaccard_overlap() &&
          jaccard_overlap < sample_constraint.min_jaccard_overlap()) {
        continue;
      }
      if (sample_constraint.has_max_jaccard_overlap() &&
          jaccard_overlap > sample_constraint.max_jaccard_overlap()) {
        continue;
      }
      found = true;
    }
    // Test sample coverage.
    if (has_sample_coverage) {
      const float sample_coverage = BBoxCoverage(sampled_bbox, object_bbox);
      if (sample_constraint.has_min_sample_coverage() &&
          sample_coverage < sample_constraint.min_sample_coverage()) {
        continue;
      }
      if (sample_constraint.has_max_sample_coverage() &&
          sample_coverage > sample_constraint.max_sample_coverage()) {
        continue;
      }
      found = true;
    }
    // Test object coverage.
    if (has_object_coverage) {
      const float object_coverage = BBoxCoverage(object_bbox, sampled_bbox);
      if (sample_constraint.has_min_object_coverage() &&
          object_coverage <= sample_constraint.min_object_coverage()) {
        continue;
      }
      if (sample_constraint.has_max_object_coverage() &&
          object_coverage > sample_constraint.max_object_coverage()) {
        continue;
      }
      found = true;
    }
    if (found) {
      return true;
    }
  }
  return found;
}

void SampleBBox(const Sampler& sampler, NormalizedBBox* sampled_bbox) {
  // Get random scale.
  CHECK_GE(sampler.max_scale(), sampler.min_scale());
  CHECK_GT(sampler.min_scale(), 0.);
  CHECK_LE(sampler.max_scale(), 1.);
  float scale;
  caffe_rng_uniform(1, sampler.min_scale(), sampler.max_scale(), &scale);

  // Get random aspect ratio.
  CHECK_GE(sampler.max_aspect_ratio(), sampler.min_aspect_ratio());
  CHECK_GT(sampler.min_aspect_ratio(), 0.);
  CHECK_LT(sampler.max_aspect_ratio(), FLT_MAX);
  float aspect_ratio;
  caffe_rng_uniform(1, sampler.min_aspect_ratio(), sampler.max_aspect_ratio(),
      &aspect_ratio);

  aspect_ratio = std::max<float>(aspect_ratio, std::pow(scale, 2.));
  aspect_ratio = std::min<float>(aspect_ratio, 1 / std::pow(scale, 2.));

  // Figure out bbox dimension.
  float bbox_width = scale * sqrt(aspect_ratio);
  float bbox_height = scale / sqrt(aspect_ratio);

  // Figure out top left coordinates.
  float w_off, h_off;
  caffe_rng_uniform(1, 0.f, 1 - bbox_width, &w_off);
  caffe_rng_uniform(1, 0.f, 1 - bbox_height, &h_off);

  sampled_bbox->set_xmin(w_off);
  sampled_bbox->set_ymin(h_off);
  sampled_bbox->set_xmax(w_off + bbox_width);
  sampled_bbox->set_ymax(h_off + bbox_height);
}

void GenerateSamples(const NormalizedBBox& source_bbox,
                     const vector<NormalizedBBox>& object_bboxes,
                     const BatchSampler& batch_sampler,
                     vector<NormalizedBBox>* sampled_bboxes) {
  int found = 0;
  for (int i = 0; i < batch_sampler.max_trials(); ++i) {
    if (batch_sampler.has_max_sample() &&
        found >= batch_sampler.max_sample()) {
      break;
    }
    // Generate sampled_bbox in the normalized space [0, 1].
    NormalizedBBox sampled_bbox;
    SampleBBox(batch_sampler.sampler(), &sampled_bbox);
    // Transform the sampled_bbox w.r.t. source_bbox.
    LocateBBox(source_bbox, sampled_bbox, &sampled_bbox);
    // Determine if the sampled bbox is positive or negative by the constraint.
    if (SatisfySampleConstraint(sampled_bbox, object_bboxes,
                                batch_sampler.sample_constraint())) {
      ++found;
      sampled_bboxes->push_back(sampled_bbox);
    }
  }
}

void GenerateBatchSamples(const AnnotatedDatum& anno_datum,
                          const vector<BatchSampler>& batch_samplers,
                          vector<NormalizedBBox>* sampled_bboxes) {
  sampled_bboxes->clear();
  vector<NormalizedBBox> object_bboxes;
  GroupObjectBBoxes(anno_datum, &object_bboxes);
  for (int i = 0; i < batch_samplers.size(); ++i) {
    if (batch_samplers[i].use_original_image()) {
      NormalizedBBox unit_bbox;
      unit_bbox.set_xmin(0);
      unit_bbox.set_ymin(0);
      unit_bbox.set_xmax(1);
      unit_bbox.set_ymax(1);
      GenerateSamples(unit_bbox, object_bboxes, batch_samplers[i],
                      sampled_bboxes);
    }
  }
}

float compareMin(float value_a, float value_b){
  if(value_a >= value_b)
    return value_b;
  else
    return value_a;
}

void GenerateDataAnchorSample(const AnnotatedDatum& anno_datum, 
                                const DataAnchorSampler& data_anchor_sampler,
                                const vector<NormalizedBBox>& object_bboxes,
                                int resized_height, int resized_width,
                                NormalizedBBox* samplerbox){
  vector<int>anchorScale;
  int img_height = anno_datum.datum().height();
  int img_width = anno_datum.datum().width();
  anchorScale.clear();
  for(int s = 0 ; s < data_anchor_sampler.scale_size(); s++){
    anchorScale.push_back(data_anchor_sampler.scale(s));
  }
  CHECK_GT(object_bboxes.size(), 0);
  int object_bbox_index = caffe_rng_rand() % object_bboxes.size();
  const float xmin = object_bboxes[object_bbox_index].xmin()*img_width;
  const float xmax = object_bboxes[object_bbox_index].xmax()*img_width;
  const float ymin = object_bboxes[object_bbox_index].ymin()*img_height;
  const float ymax = object_bboxes[object_bbox_index].ymax()*img_height;
  float bbox_width = xmax - xmin;
  float bbox_height = ymax - ymin;
  int anchor_range_size = 0, anchor_choose_index = 0, rng_random_index = 0; 
  float bbox_aera = bbox_height * bbox_width;
  float scaleChoose = 0.0f; 
  for(int j = 0; j < anchorScale.size() -1; ++j){
    if(bbox_aera >= std::pow(anchorScale[j], 2) && bbox_aera < std::pow(anchorScale[j+1], 2))
    {
      if(std::fabs(bbox_aera - std::pow(anchorScale[j], 2)) <= std::fabs(bbox_aera - std::pow(anchorScale[j + 1], 2))){
        anchor_range_size = j;
      }else{
        anchor_range_size = j + 1;
      }
      break;
    }
  }
  if(anchor_range_size==0){
    anchor_choose_index = 0;
  }else{
    rng_random_index = caffe_rng_rand() % (anchor_range_size + 1);
    anchor_choose_index = rng_random_index % (anchor_range_size + 1);
  }
  if(anchor_choose_index == anchor_range_size){
    float min_resize_val = anchorScale[anchor_choose_index] / 2;
    float max_resize_val = compareMin((float)anchorScale[anchor_choose_index] * 2,
                                                  2*std::sqrt(bbox_aera)) ;
    caffe_rng_uniform(1, min_resize_val, max_resize_val, &scaleChoose);
  }else{
    float min_resize_val = anchorScale[anchor_choose_index] / 2;
    float max_resize_val = (float)anchorScale[anchor_choose_index] * 2;
    caffe_rng_uniform(1, min_resize_val, max_resize_val, &scaleChoose);
  }
  float sample_box_size = (float)bbox_width * resized_width / scaleChoose;

  float width_offset_org = 0.0f, height_offset_org = 0.0f;
  if(sample_box_size < std::max(img_width, img_height)){
    if(bbox_width <= sample_box_size){
      caffe_rng_uniform(1, bbox_width + xmin - sample_box_size, xmin, &width_offset_org );
    }else{
      caffe_rng_uniform(1, xmin, bbox_width + xmin - sample_box_size, &width_offset_org);
    }
    if(bbox_height <= sample_box_size){
      caffe_rng_uniform(1, ymin + bbox_height - sample_box_size, ymin, &height_offset_org);
    }else{
      caffe_rng_uniform(1, ymin, ymin + bbox_height - sample_box_size, &height_offset_org);
    }
  }else{
    caffe_rng_uniform(1, img_height-sample_box_size, 0.0f, &height_offset_org);
    caffe_rng_uniform(1, img_width-sample_box_size, 0.0f, &width_offset_org);
  }
  int width_offset_ = std::floor(width_offset_org);
  int height_offset_ = std::floor(height_offset_org);
  float w_off = (float) width_offset_ / img_width;
  float h_off = (float) height_offset_ / img_height;
  samplerbox->set_xmin(w_off);
  samplerbox->set_ymin(h_off);
  samplerbox->set_xmax(w_off + float(sample_box_size/img_width));
  samplerbox->set_ymax(h_off + float(sample_box_size/img_height));
  NormalizedBBox source_bbox;
  source_bbox.set_xmin(0);
  source_bbox.set_ymin(0);
  source_bbox.set_xmax(1);
  source_bbox.set_ymax(1);
  LocateBBox(source_bbox, *samplerbox, samplerbox);
  
}// func GenerateDataAnchorSamples

void GenerateBatchDataAnchorSamples(const AnnotatedDatum& anno_datum,
                                const vector<DataAnchorSampler>& data_anchor_samplers,
                                int resized_height, int resized_width, 
                                vector<NormalizedBBox>* sampled_bboxes) {
  sampled_bboxes->clear();
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
        NormalizedBBox samplebox;
        GenerateDataAnchorSample(anno_datum, data_anchor_samplers[i], object_bboxes, resized_height, 
                                resized_width, &samplebox);
        if (SatisfySampleConstraint(samplebox, object_bboxes,
                                      data_anchor_samplers[i].sample_constraint())){
          found++;
          sampled_bboxes->push_back(samplebox);
        }
      }
    }
  }
}

}  // namespace caffe
