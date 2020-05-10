#include <algorithm>
#include <csignal>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/sampler.hpp"
#include "caffe/data_transformer.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/annotated_data_layer.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
using namespace cv;
using namespace std;

#define TEST_CROP_BATCH 0
#define TEST_CROP_BOX 0
#define TEST_CROP_ANCHOR 1
#define TEST_CROP_JITTER 0


int main(int argc, char** argv){
    int loop_time = 50;
    int batch_size = 64;
    int Resized_Height = 640;
    int Resized_Width = 640;
    std::string srcTestfile = "../../img_test.txt";
    std::string save_folder = "../../anchorTestImage";
    std::vector<std::pair<string, string> > img_filenames;
    // anchor samples policy
    // anchor 的大概范围16, 32, 64, 128, 256, 512
    std::vector<int> Anchors;
    for(unsigned nn = 0; nn < 6; nn++){
        Anchors.push_back(16 * std::pow(2, nn));
    }
    std::vector<std::vector<int> >anchorSamples;
    anchorSamples.push_back(Anchors);
    // 设计LFFD-BOX 方式的Crop采样参数
    // low gt_boxes_list: 10, 15, 20, 40, 70, 110, 250, 400
    // up gt_boxes_list: 15, 20, 40, 70, 110, 250, 400, 560
    // anchorStride_list: 4, 4, 8, 8, 16, 32, 32, 32
    int low_boxes_list[8] = { 10, 15, 20, 40, 70, 110, 250, 400};
    int up_boxes_list[8] = {15, 20, 40, 70, 110, 250, 400, 560};
    int anchorStride_list[8] = {4, 4, 8, 8, 16, 32, 32, 32};
    std::vector<int> low_gt_boxes_list, up_gt_boxes_list, anchor_stride_list;
    for (size_t i = 0; i < 8; i++)
    {
        low_gt_boxes_list.push_back(low_boxes_list[i]);
        up_gt_boxes_list.push_back(up_boxes_list[i]);
        anchor_stride_list.push_back(anchorStride_list[i]);
    }
    // 设置采样参数 data_anchor
    vector<DataAnchorSampler> data_anchor_samplers_;
    for(int ii = 0; ii < 1; ii++){
    DataAnchorSampler samplers;
    for(int jj = 0; jj < 6; jj++){
        samplers.add_scale(16 * int(std::pow(2, jj)));
    }
    samplers.mutable_sample_constraint()->set_min_object_coverage(0.5);
    samplers.set_max_sample(1);
    samplers.set_max_trials(10);
    data_anchor_samplers_.push_back(samplers);
    }

    // 设置CROP BATCH 采样参数
    vector<BatchSampler> batch_samplers;
    for(int ii = 0; ii < 7; ii++){
        BatchSampler batch_sampler;
        batch_sampler.set_max_trials(50);
        batch_sampler.set_max_sample(1);
        float overlap = (0.65 + ii * 0.1)?(0.65 + ii * 0.1) <= 1 : 1;
        batch_sampler.mutable_sample_constraint()->set_min_object_coverage(overlap);
        float ratio = (0.3 + ii * 0.1)?(0.3 + ii * 0.1) <= 1 : 1;
        batch_sampler.mutable_sampler()->set_min_scale(ratio);
        batch_sampler.mutable_sampler()->set_max_scale(1.0);
        batch_sampler.mutable_sampler()->set_min_aspect_ratio(ratio);
        batch_sampler.mutable_sampler()->set_max_aspect_ratio(1.0);
        batch_samplers.push_back(batch_sampler);
    }

    // 读文件，文件里面存着真实值，包括图像文件， 和真是坐标值文件
    // 随机裁剪，生成再Resize到相对应大小的（640， 640）
    std::ifstream infile(srcTestfile.c_str());
    string line;
    size_t pos;
    std::stringstream sstr ;
    while(std::getline(infile, line)){
        pos = line.find_last_of(' ');
        std::string label_file = line.substr(pos+1);
        std::string img_file = line.substr(0, pos);
        img_filenames.push_back(std::make_pair(img_file, label_file));
    }
    infile.close();
    int numSamples = img_filenames.size();
    // 生成datum，再利用正宗的数据增强接口， 再将其转化为图像
    std::vector<AnnotatedDatum> source_datum;
    std::map<std::string, int> name_to_label;
    const bool check_label = true;
    const string label_map_file = "../examples/face/detector/labelmap.prototxt";
    LabelMap label_map;
    CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
        << "Failed to read label map file.";
    CHECK(MapNameToLabel(label_map, check_label, &name_to_label))
        << "Failed to convert name to label.";
    const string label_type = "xml";
    const int resize_height = 0;
    const int resize_width = 0;
    const int min_dim = 0;
    const int max_dim = 0;
    const bool is_color = true;
    AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX;
    std::string encode_type = "jpg";
    for(int ii = 0; ii < numSamples; ii++){
    std::string filename = img_filenames[ii].first;
    std::string labelname = img_filenames[ii].second;

    AnnotatedDatum anno_datum;
    ReadRichImageToAnnotatedDatum(filename, labelname, resize_height,
        resize_width, min_dim, max_dim, is_color, encode_type, type, label_type,
        name_to_label, &anno_datum);
    anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
    source_datum.push_back(anno_datum);
    }
    LOG(INFO)<<"=====SOURCE DATUM SUCCESSFULLY!=====";

    // 生成Datatransfrm的参数
    TransformationParameter transform_param;
    transform_param.set_mirror(false);
    ResizeParameter* resized_param = transform_param.mutable_resize_param();
    resized_param->set_height(Resized_Height);
    resized_param->set_width(Resized_Width);
    resized_param->set_resize_mode(ResizeParameter_Resize_mode_WARP);
    resized_param->set_prob(1.0);
    resized_param->add_interp_mode(ResizeParameter_Interp_mode_LINEAR);
    resized_param->add_interp_mode(ResizeParameter_Interp_mode_AREA);
    resized_param->add_interp_mode(ResizeParameter_Interp_mode_NEAREST);
    resized_param->add_interp_mode(ResizeParameter_Interp_mode_CUBIC);
    resized_param->add_interp_mode(ResizeParameter_Interp_mode_LANCZOS4);

    EmitConstraint* emit_constranit_ = transform_param.mutable_emit_constraint();
    emit_constranit_->set_emit_type(EmitConstraint_EmitType_CENTER);

    DataTransformer<float> data_transformer_(transform_param, TEST);
    // 循环操作
    for(int ii = 0; ii < loop_time; ii++){
        for(int jj = 0; jj < batch_size; jj++){
            int rand_idx = caffe_rng_rand() % numSamples;
            int pos_name = img_filenames[rand_idx].first.find_last_of("/");
            std::string img_name = img_filenames[rand_idx].first.substr(pos_name);
            int pos_suffix = img_name.find_last_of(".");
            std::string prefix_imgName = img_name.substr(0, pos_suffix);

            AnnotatedDatum& anno_datum = source_datum[rand_idx];
            AnnotatedDatum* sampled_datum = NULL;
            AnnotatedDatum* resized_anno_datum = NULL;
            bool do_resize = true;
            if(do_resize)
            resized_anno_datum = new AnnotatedDatum();
            vector<NormalizedBBox> sampled_bboxes;
            #if TEST_CROP_ANCHOR
            GenerateBatchDataAnchorSamples(anno_datum, data_anchor_samplers_, &sampled_bboxes);
            sampled_datum = new AnnotatedDatum();
            int rand_id = caffe_rng_rand() % sampled_bboxes.size();
            data_transformer_.CropImage(anno_datum,
                                        sampled_bboxes[rand_id],
                                        sampled_datum);
            LOG(INFO)<<"=====TEST DATA ANCHOR SAMPLES SUCCESSFULLY!=====";
            #endif
            #if TEST_CROP_BOX
            GenerateLFFDSample(anno_datum, &sampled_bboxes, 
                                    low_gt_boxes_list, up_gt_boxes_list, anchor_stride_list,
                                    resized_anno_datum, transform_param, do_resize);
            CHECK_GT(resized_anno_datum->datum().channels(), 0);
            sampled_datum = new AnnotatedDatum();
            int rand_id = caffe_rng_rand() % sampled_bboxes.size();
            data_transformer_.CropImage_Sampling(*resized_anno_datum,
                                                sampled_bboxes[rand_id],
                                                sampled_datum);
            LOG(INFO)<<"=====TEST DATA LFFD SAMPLES SUCCESSFULLY!=====";
            #endif
            #if TEST_CROP_BATCH
            GenerateBatchSamples(anno_datum, batch_samplers, &sampled_bboxes);
            sampled_datum = new AnnotatedDatum();
            int rand_id = caffe_rng_rand() % sampled_bboxes.size();
            data_transformer_.CropImage(anno_datum,
                                        sampled_bboxes[rand_id],
                                        sampled_datum);
            #endif
            #if TEST_CROP_JITTER
            GenerateJitterSamples(anno_datum, 0.3, &sampled_bboxes);
            sampled_datum = new AnnotatedDatum();
            int rand_id = caffe_rng_rand() % sampled_bboxes.size();
            data_transformer_.CropImage(anno_datum,
                                        sampled_bboxes[rand_id],
                                        sampled_datum);
            #endif
            CHECK(sampled_datum != NULL);
            sampled_datum->set_type(AnnotatedDatum_AnnotationType_BBOX);
            cv::Mat cropImage;
            std::string saved_img_name = save_folder + "/" + prefix_imgName + "_" + to_string(ii) + "_" + to_string(jj) +".jpg";
            #if 1
            vector<AnnotationGroup> transformed_anno_vec;
            transformed_anno_vec.clear();
            Blob<float> transformed_blob;
            vector<int> shape = data_transformer_.InferBlobShape(sampled_datum->datum());
            for(unsigned int ii = 0; ii < shape.size(); ii++){
            std::cout <<shape[ii]<<std::endl;
            }
            transformed_blob.Reshape(shape);
            data_transformer_.Transform(*sampled_datum, &transformed_blob, &transformed_anno_vec);
            LOG(INFO)<<"SAMPEL SUCCESSFULLY";
            // 将Datum数据转换为原来图像的数据, 并保存成图像
            cropImage = cv::Mat(transformed_blob.height(), transformed_blob.width(), CV_8UC3);
            const float* data = transformed_blob.cpu_data();
            int Trans_Height = transformed_blob.height();
            int Trans_Width = transformed_blob.width();
            for(int row = 0; row < Trans_Height; row++){
            unsigned char *ImgData = cropImage.ptr<uchar>(row);
            for(int col = 0; col < Trans_Width; col++){
                ImgData[3 * col + 0] = static_cast<uchar>(data[0 * Trans_Height * Trans_Width + row * Trans_Width + col]);
                ImgData[3 * col + 1] = static_cast<uchar>(data[1 * Trans_Height * Trans_Width + row * Trans_Width + col]);
                ImgData[3 * col + 2] = static_cast<uchar>(data[2 * Trans_Height * Trans_Width + row * Trans_Width + col]);
            }
        }
        int Crop_Height = cropImage.rows;
        int Crop_Width = cropImage.cols;
        for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = transformed_anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
                const Annotation& anno = anno_group.annotation(a);
                const NormalizedBBox& bbox = anno.bbox();
                int xmin = int(bbox.xmin() * Crop_Width);
                int ymin = int(bbox.ymin() * Crop_Height);
                int xmax = int(bbox.xmax() * Crop_Width);
                int ymax = int(bbox.ymax() * Crop_Height);
                cv::rectangle(cropImage, cv::Point2i(xmin, ymin), cv::Point2i(xmax, ymax), cv::Scalar(255,0,0), 1, 1, 0);
            }
        }
        #else
        cropImage= DecodeDatumToCVMatNative(sampled_datum->datum());
        int Crop_Height = cropImage.rows;
        int Crop_Width = cropImage.cols;
        for (int g = 0; g < sampled_datum->annotation_group_size(); ++g) {
                const AnnotationGroup& anno_group = sampled_datum->annotation_group(g);
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
                const Annotation& anno = anno_group.annotation(a);
                const NormalizedBBox& bbox = anno.bbox();
                int xmin = int(bbox.xmin() * Crop_Width);
                int ymin = int(bbox.ymin() * Crop_Height);
                int xmax = int(bbox.xmax() * Crop_Width);
                int ymax = int(bbox.ymax() * Crop_Height);
                cv::rectangle(cropImage, cv::Point2i(xmin, ymin), cv::Point2i(xmax, ymax), cv::Scalar(255,0,0), 1, 1, 0);
            }
        }
        #endif
        
        cv::imwrite(saved_img_name, cropImage);
        LOG(INFO)<<"*** Datum Write Into Jpg File Sucessfully! ***";
        delete sampled_datum;
        if(do_resize)
        delete resized_anno_datum;
    }
    }
	return 1;
}
