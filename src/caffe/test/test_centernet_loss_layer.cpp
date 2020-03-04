#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/centernet_object_loss_layer.hpp"
#include "caffe/layers/permute_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/prior_box_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

static bool kBoolChoices[] = {true, false};
static CenterObjectParameter_LocLossType kLocLossTypes[] = {
  CenterObjectParameter_LocLossType_L2,
  CenterObjectParameter_LocLossType_SMOOTH_L1};
static CenterObjectParameter_ConfLossType kConfLossTypes[] = {
  CenterObjectParameter_ConfLossType_FOCALSIGMOID};
static LossParameter_NormalizationMode kNormalizationModes[] = {
  LossParameter_NormalizationMode_BATCH_SIZE,
  LossParameter_NormalizationMode_FULL,
  LossParameter_NormalizationMode_VALID,
  LossParameter_NormalizationMode_NONE};

template <typename TypeParam>
class centerNetLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  centerNetLossLayerTest()
      : num_(3),
        num_classes_(3),
        width_(20),
        height_(20),
        blob_bottom_loc_(new Blob<Dtype>(num_, 2, 20, 20)),
        blob_bottom_wh_(new Blob<Dtype>(num_, 2, 20, 20)),
        blob_bottom_conf_(new Blob<Dtype>(num_, num_classes_, 20, 20)),
        blob_bottom_gt_(new Blob<Dtype>(1, 1, 4, 8)),
        blob_top_loss_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_loc_);
    blob_bottom_vec_.push_back(blob_bottom_conf_);
    blob_bottom_vec_.push_back(blob_bottom_wh_);
    blob_bottom_vec_.push_back(blob_bottom_gt_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~centerNetLossLayerTest() {
    delete blob_bottom_loc_;
    delete blob_bottom_wh_;
    delete blob_bottom_conf_;
    delete blob_bottom_gt_;
    delete blob_top_loss_;
  }

  void FillItem(Dtype* blob_data, const string values) {
    // Split values to vector of items.
    vector<string> items;
    std::istringstream iss(values);
    std::copy(std::istream_iterator<string>(iss),
              std::istream_iterator<string>(), back_inserter(items));
    int num_items = items.size();
    CHECK_EQ(num_items, 8);

    for (int i = 0; i < 8; ++i) {
      if (i >= 3 && i <= 6) {
        blob_data[i] = atof(items[i].c_str());
      } else {
        blob_data[i] = atoi(items[i].c_str());
      }
    }
  }

  // Fill the bottom blobs.
  void Fill(bool share_location) {
    int loc_classes = share_location ? 1 : num_classes_;
    // Create fake network which simulates a simple multi box network.
    vector<Blob<Dtype>*> fake_bottom_vec;
    vector<Blob<Dtype>*> fake_top_vec;
    LayerParameter layer_param;
    // Fake input (image) of size 20 x 20
    Blob<Dtype>* fake_input = new Blob<Dtype>(num_, 3, 20, 20);

    // 0) Fill ground truth.
#ifdef USE_LMDB
    string filename;
    GetTempDirname(&filename);
    DataParameter_DB backend = DataParameter_DB_LMDB;
    scoped_ptr<db::DB> db(db::GetDB(backend));
    db->Open(filename, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());
    for (int i = 0; i < num_; ++i) {
      AnnotatedDatum anno_datum;
      // Fill data.
      Datum* datum = anno_datum.mutable_datum();
      datum->set_channels(3);
      datum->set_height(20);
      datum->set_width(20);
      std::string* data = datum->mutable_data();
      for (int j = 0; j < 3*20*20; ++j) {
        data->push_back(static_cast<uint8_t>(j/100.));
      }
      anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
      if (i == 0 || i == 2) {
        AnnotationGroup* anno_group = anno_datum.add_annotation_group();
        anno_group->set_group_label(1);
        Annotation* anno = anno_group->add_annotation();
        anno->set_instance_id(0);
        NormalizedBBox* bbox = anno->mutable_bbox();
        bbox->set_xmin(0.1);
        bbox->set_ymin(0.1);
        bbox->set_xmax(0.3);
        bbox->set_ymax(0.3);
        bbox->set_difficult(i % 2);
      }
      if (i == 2) {
        AnnotationGroup* anno_group = anno_datum.add_annotation_group();
        anno_group->set_group_label(2);
        Annotation* anno = anno_group->add_annotation();
        anno->set_instance_id(0);
        NormalizedBBox* bbox = anno->mutable_bbox();
        bbox->set_xmin(0.2);
        bbox->set_ymin(0.2);
        bbox->set_xmax(0.4);
        bbox->set_ymax(0.4);
        bbox->set_difficult(i % 2);
        anno = anno_group->add_annotation();
        anno->set_instance_id(1);
        bbox = anno->mutable_bbox();
        bbox->set_xmin(0.6);
        bbox->set_ymin(0.6);
        bbox->set_xmax(0.8);
        bbox->set_ymax(0.9);
        bbox->set_difficult((i + 1) % 2);
      }
      string key_str = caffe::format_int(i, 3);
      string out;
      CHECK(anno_datum.SerializeToString(&out));
      txn->Put(key_str, out);
    }
    txn->Commit();
    db->Close();
    DataParameter* data_param = layer_param.mutable_data_param();
    data_param->set_batch_size(num_);
    data_param->set_source(filename.c_str());
    data_param->set_backend(backend);
    AnnotatedDataLayer<Dtype> anno_data_layer(layer_param);
    fake_top_vec.clear();
    fake_top_vec.push_back(fake_input);
    fake_top_vec.push_back(blob_bottom_gt_);
    anno_data_layer.SetUp(fake_bottom_vec, fake_top_vec);
    anno_data_layer.Forward(fake_bottom_vec, fake_top_vec);
#else
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(fake_input);
    vector<int> gt_shape(4, 1);
    gt_shape[2] = 4;
    gt_shape[3] = 8;
    blob_bottom_gt_->Reshape(gt_shape);
    Dtype* gt_data = blob_bottom_gt_->mutable_cpu_data();
    FillItem(gt_data, "0 1 0 0.1 0.1 0.3 0.3 0");
    FillItem(gt_data + 8, "2 1 0 0.1 0.1 0.3 0.3 0");
    FillItem(gt_data + 8 * 2, "2 2 0 0.2 0.2 0.4 0.4 0");
    FillItem(gt_data + 8 * 3, "2 2 1 0.6 0.6 0.8 0.9 1");
#endif  // USE_LMDB

    // Fake layer
    // 1) Fill bbox location predictions.
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_pad(0);
    convolution_param->add_kernel_size(1);
    convolution_param->add_stride(1);
    int num_output = loc_classes * 2;
    convolution_param->set_num_output(num_output);
    convolution_param->mutable_weight_filler()->set_type("xavier");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0.1);
    ConvolutionLayer<Dtype> conv_layer_loc(layer_param);
    fake_bottom_vec.clear();
    fake_bottom_vec.push_back(fake_input);
    Blob<Dtype> fake_output_loc;
    fake_top_vec.clear();
    fake_top_vec.push_back(&fake_output_loc);
    conv_layer_loc.SetUp(fake_bottom_vec, fake_top_vec);
    conv_layer_loc.Forward(fake_bottom_vec, fake_top_vec);

    // 2) Fill bbox wh size  predictions.

    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_pad(0);
    convolution_param->add_kernel_size(1);
    convolution_param->add_stride(1);
    int num_output = loc_classes * 2;
    convolution_param->set_num_output(num_output);
    convolution_param->mutable_weight_filler()->set_type("xavier");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0.1);
    ConvolutionLayer<Dtype> conv_layer_wh(layer_param);
    fake_bottom_vec.clear();
    fake_bottom_vec.push_back(fake_input);
    Blob<Dtype> fake_output_loc;
    fake_top_vec.clear();
    fake_top_vec.push_back(&fake_output_loc);
    conv_layer_wh.SetUp(fake_bottom_vec, fake_top_vec);
    conv_layer_wh.Forward(fake_bottom_vec, fake_top_vec);

    // 3) Fill bbox confidence predictions.
    convolution_param->set_num_output(num_classes_);
    ConvolutionLayer<Dtype> conv_layer_conf(layer_param);
    fake_bottom_vec.clear();
    fake_bottom_vec.push_back(fake_input);
    num_output = num_classes_;
    Blob<Dtype> fake_output_conf;
    fake_top_vec.clear();
    fake_top_vec.push_back(&fake_output_conf);
    conv_layer_conf.SetUp(fake_bottom_vec, fake_top_vec);
    conv_layer_conf.Forward(fake_bottom_vec, fake_top_vec);

    fake_bottom_vec.clear();
    fake_bottom_vec.push_back(&fake_output_conf);
    fake_top_vec.clear();
    
    delete fake_input;
  }
  int num_;
  int num_classes_;
  int width_;
  int height_;
  Blob<Dtype>* const blob_bottom_loc_;
  Blob<Dtype>* const blob_bottom_conf_;
  Blob<Dtype>* const blob_bottom_wh_;
  Blob<Dtype>* const blob_bottom_gt_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(centerNetLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(centerNetLossLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CenterObjectParameter* center_object_loss_param =
      layer_param.mutable_center_object_loss_param();
  center_object_loss_param->set_num_class(3);
  for (int i = 0; i < 2; ++i) {
    bool share_location = kBoolChoices[i];
    this->Fill(share_location);
    center_object_loss_param->set_share_location(share_location);
    CenterObjectLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  }
}

TYPED_TEST(centerNetLossLayerTest, TestLocGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(false);
  LossParameter* loss_param = layer_param.mutable_loss_param();
  CenterObjectParameter* center_object_loss_param =
      layer_param.mutable_center_object_loss_param();
  center_object_loss_param->set_num_class(this->num_classes_);
  for (int l = 0; l < 2; ++l) {
    CenterObjectParameter_LocLossType loc_loss_type = kLocLossTypes[l];
    for (int i = 0; i < 2; ++i) {
      bool share_location = kBoolChoices[i];
      this->Fill(share_location);
      for (int n = 0; n < 4; ++n) {
        LossParameter_NormalizationMode normalize = kNormalizationModes[n];
        loss_param->set_normalization(normalize);
        center_object_loss_param->set_loc_loss_type(loc_loss_type);
        center_object_loss_param->set_share_location(share_location);

        center_object_loss_param->set_use_difficult_gt(true);
        CenterObjectLossLayer<Dtype> layer(layer_param);
        GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                        this->blob_top_vec_, 0);
      }
    }
  }
}

TYPED_TEST(centerNetLossLayerTest, TestConfGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LossParameter* loss_param = layer_param.mutable_loss_param();
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(true);
  CenterObjectParameter* center_object_loss_param =
      layer_param.mutable_center_object_loss_param();
  center_object_loss_param->set_num_class(this->num_classes_);
  for (int c = 0; c < 1; ++c) {
    CenterObjectParameter_ConfLossType conf_loss_type = kConfLossTypes[c];
    for (int i = 0; i < 2; ++i) {
      bool share_location = kBoolChoices[i];
      this->Fill(share_location);
      for (int n = 0; n < 4; ++n) {
        LossParameter_NormalizationMode normalize = kNormalizationModes[n];
        loss_param->set_normalization(normalize);
        for (int u = 0; u < 2; ++u) {
          bool use_difficult_gt = kBoolChoices[u];
          center_object_loss_param->set_conf_loss_type(conf_loss_type);
          center_object_loss_param->set_share_location(share_location);
          center_object_loss_param->set_use_difficult_gt(use_difficult_gt);
          center_object_loss_param->set_background_label_id(0);
          CenterObjectLossLayer<Dtype> layer(layer_param);
          GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
          checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                          this->blob_top_vec_, 1);
        }
      }
    }
  }
}

}  // namespace caffe
