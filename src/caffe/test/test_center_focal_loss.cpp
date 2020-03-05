#include <cmath>
#include <vector>
#include <algorithm>
#include <cfloat>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/focal_loss_layer_centernet.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/center_bbox_util.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CenterNetfocalSigmoidWithLossLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CenterNetfocalSigmoidWithLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(5, 1, 80, 80)),
        blob_bottom_targets_(new Blob<Dtype>(5, 1, 80, 80)),
        blob_top_loss_(new Blob<Dtype>()),
        alpha_(2.0), gamma_(4.0) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    // Fill the targets vector
    this->fillBottomData(5, 9);
    blob_bottom_vec_.push_back(blob_bottom_targets_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~CenterNetfocalSigmoidWithLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
    delete blob_top_loss_;
  }

  void fillBottomData(int batch_num, int total_box){
    std::map<int, vector<NormalizedBBox> > all_gt_bboxes=random_fill_gt_box(batch_num, total_box);
    Dtype* conf_gt_data = this->blob_bottom_targets_->mutable_cpu_data();
    GenerateBatchHeatmap(all_gt_bboxes, conf_gt_data, 1, 80, 80);
  }

  std::map<int, vector<NormalizedBBox> > random_fill_gt_box(int batch_num, int total_box){
    std::map<int, vector<NormalizedBBox> > all_gt_bboxes;
    int num_per = total_box / batch_num;
    int lasted_num = total_box % batch_num;
    for(int i = 0; i < batch_num; i++){
      int num = (i == batch_num -1)?(num_per + lasted_num):num_per; 
      for(int j = 0; j < num; j++){
        float bbox_width, bbox_height;
        caffe_rng_uniform(1, 0.f, 1.f, &bbox_width);
        caffe_rng_uniform(1, 0.f, 1.f, &bbox_height);

        float w_off, h_off;
        caffe_rng_uniform(1, 0.f, 1 - bbox_width, &w_off);
        caffe_rng_uniform(1, 0.f, 1 - bbox_height, &h_off);
        NormalizedBBox sampled_bbox;
        sampled_bbox.set_xmin(w_off);
        sampled_bbox.set_ymin(h_off);
        sampled_bbox.set_xmax(w_off + bbox_width);
        sampled_bbox.set_ymax(h_off + bbox_height);
        sampled_bbox.set_label(1);
        all_gt_bboxes[i].push_back(sampled_bbox);
      }
    }
    return all_gt_bboxes;
  }

  Dtype SigmoidCrossEntropyLossReference(const int count, const int num,
                                         const Dtype* input,
                                         const Dtype* target) {
    Dtype loss = 0;
    int count_ = 0;
    for (int i = 0; i < count; ++i) {
      const Dtype prediction = 1 / (1 + exp(-input[i]));
      EXPECT_LE(prediction, 1);
      EXPECT_GE(prediction, 0);
      EXPECT_LE(target[i], 1);
      EXPECT_GE(target[i], 0);
      Dtype label_a = target[i];
      if(label_a == Dtype(1.0)){
        loss -= log(std::max(prediction, Dtype(FLT_MIN))) * std::pow(1 -prediction, alpha_);       
        count_++;
      }else if(label_a < Dtype(1.0)){
        loss -= log(std::max(1 - prediction, Dtype(FLT_MIN))) * std::pow(prediction, alpha_) *
                        std::pow(1 - label_a, gamma_); 
      }
    }
    return loss / count_;
  }

  void TestForward() {
    LayerParameter layer_param;
    const Dtype kLossWeight = 1;
    layer_param.add_loss_weight(kLossWeight);
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    Dtype eps = 2e-2;
    for (int i = 0; i < 1; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the targets vector
      this->fillBottomData(5, 9);
      CenterNetfocalSigmoidWithLossLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      Dtype layer_loss =
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      const int count = this->blob_bottom_data_->count();
      const int num = this->blob_bottom_data_->num();
      const Dtype* blob_bottom_data = this->blob_bottom_data_->cpu_data();
      const Dtype* blob_bottom_targets =
          this->blob_bottom_targets_->cpu_data();
      Dtype reference_loss = kLossWeight * SigmoidCrossEntropyLossReference(
          count, num, blob_bottom_data, blob_bottom_targets);
      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Dtype alpha_;
  Dtype gamma_;
};

TYPED_TEST_CASE(CenterNetfocalSigmoidWithLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CenterNetfocalSigmoidWithLossLayerTest, TestSigmoidCrossEntropyLoss) {
  this->TestForward();
}

TYPED_TEST(CenterNetfocalSigmoidWithLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 1;
  layer_param.add_loss_weight(kLossWeight);
  CenterNetfocalSigmoidWithLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
