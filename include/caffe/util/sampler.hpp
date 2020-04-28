#ifndef CAFFE_UTIL_SAMPLER_H_
#define CAFFE_UTIL_SAMPLER_H_

#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

namespace caffe {

void GenerateJitterSamples(const AnnotatedDatum& anno_datum, float jitter, vector<NormalizedBBox>* sampled_bboxes);

// Find all annotated NormalizedBBox.
void GroupObjectBBoxes(const AnnotatedDatum& anno_datum,
                       vector<NormalizedBBox>* object_bboxes);

// Check if a sampled bbox satisfy the constraints with all object bboxes.
bool SatisfySampleConstraint(const NormalizedBBox& sampled_bbox,
                             const vector<NormalizedBBox>& object_bboxes,
                             const SampleConstraint& sample_constraint);

// Sample a NormalizedBBox given the specifictions.
void SampleBBox(const Sampler& sampler, NormalizedBBox* sampled_bbox, float orl_ratio);
void SampleBBox_Square(const AnnotatedDatum& anno_datum, const Sampler& sampler, 
                                NormalizedBBox* sampled_bbox);

// Generate samples from NormalizedBBox using the BatchSampler.
void GenerateSamples(const NormalizedBBox& source_bbox,
                     const vector<NormalizedBBox>& object_bboxes,
                     const BatchSampler& batch_sampler,
                     vector<NormalizedBBox>* sampled_bboxes, float orl_ratio);

void GenerateSamples_Square(const AnnotatedDatum& anno_datum,
                     const NormalizedBBox& source_bbox,
                     const vector<NormalizedBBox>& object_bboxes,
                     const BatchSampler& batch_sampler,
                     vector<NormalizedBBox>* sampled_bboxes);


// Generate samples from AnnotatedDatum using the BatchSampler.
// All sampled bboxes which satisfy the constraints defined in BatchSampler
// is stored in sampled_bboxes.
void GenerateBatchSamples(const AnnotatedDatum& anno_datum,
                          const vector<BatchSampler>& batch_samplers,
                          vector<NormalizedBBox>* sampled_bboxes);

void GenerateBatchSamples_Square(const AnnotatedDatum& anno_datum,
                          const vector<BatchSampler>& batch_samplers,
                          vector<NormalizedBBox>* sampled_bboxes);

// Generate samples by using data_anchor_samples
// all sampled boxes which satisfy the constraints defined in DataAnchorSampler
// is stored in sampled bboxes.
void GenerateDataAnchorSample(const AnnotatedDatum& anno_datum, 
                                const DataAnchorSampler& data_anchor_sampler,
                                const vector<NormalizedBBox>& object_bboxes,
                                int resized_height, int resized_width,
                                NormalizedBBox* sampled_bbox, 
                                const TransformationParameter& trans_param);

void GenerateBatchDataAnchorSamples(const AnnotatedDatum& anno_datum,
                                const vector<DataAnchorSampler>& data_anchor_samplers,
                                int resized_height, int resized_width,
                                NormalizedBBox* sampled_bbox,
                                const TransformationParameter& trans_param);

void ResizedCropSample(const AnnotatedDatum& anno_datum, AnnotatedDatum* resized_anno_datum, 
                       float scale, const TransformationParameter& trans_param);

// LFFD Generate samples by using data_samples

void GenerateLFFDSample(const AnnotatedDatum& anno_datum,
                        int resized_height, int resized_width,
                        NormalizedBBox* sampled_bbox, 
                        std::vector<int> bbox_small_size_list,
                        std::vector<int> bbox_large_size_list,
                        std::vector<int> anchorStride,
                        AnnotatedDatum* resized_anno_datum, 
                        const TransformationParameter& trans_param,
                        bool do_resize);


}  // namespace caffe

#endif  // CAFFE_UTIL_SAMPLER_H_
