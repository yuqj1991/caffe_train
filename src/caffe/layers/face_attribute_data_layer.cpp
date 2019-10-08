#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/face_attribute_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe{

template <typename Dtype>
faceAttributeDataLayer<Dtype>::faceAttributeDataLayer(const LayerParameter & param)
    : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param){
}

template <typename Dtype>
faceAttributeDataLayer<Dtype>::~faceAttributeDataLayer(){
    this->StopInternalThread();
}

template <typename Dtype>
void faceAttributeDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    const int batch_size = this->layer_param_.data_param().batch_size();
    // Make sure dimension is consistent within batch.
    const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
    if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
    }
    // Read a data point, and use it to initialize the top blob.
    AnnoFaceAttributeDatum& anno_datum = *(reader_.full().peek());

    // Use data_transformer to infer the expected blob shape from anno_datum.
    vector<int> top_shape =
        this->data_transformer_->InferBlobShape(anno_datum.datum());
    this->transformed_data_.Reshape(top_shape);
    // Reshape top[0] and prefetch_data according to the batch_size.
    top_shape[0] = batch_size;
    top[0]->Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].data_.Reshape(top_shape);
    }
    LOG(INFO) << "output data size: " << top[0]->num() << ","
        << top[0]->channels() << "," << top[0]->height() << ","
        << top[0]->width();
    // label
    if (this->output_labels_) {
        has_anno_type_ = anno_datum.has_type();
        vector<int> label_shape(4, 1);
        if (has_anno_type_) {
            anno_type_ = anno_datum.type();
            if (anno_type_ == AnnoFaceAttributeDatum_AnnoType_FACEATTRIBUTE) {
            label_shape[0] = 1;
            label_shape[1] = 1;
            // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
            // cpu_data and gpu_data for consistent prefetch thread. Thus we make
            // sure there is at least one bbox.
            label_shape[2] = batch_size;
            label_shape[3] = 21;
            } else {
            LOG(FATAL) << "Unknown annotation type.";
            }
        } else {
            label_shape[0] = batch_size;
        }
        top[1]->Reshape(label_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(label_shape);
        }
    }
}

// This function is called on prefetch thread
template<typename Dtype>
void faceAttributeDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    // Reshape according to the first datum of each batch
    // on single input batches allows for inputs of varying dimension.
    const int batch_size = this->layer_param_.data_param().batch_size();
    const TransformationParameter& transform_param = this->layer_param_.transform_param();
    AnnoFaceAttributeDatum& anno_datum = *(reader_.full().peek());
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(anno_datum.datum());
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);

    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

      // Store transformed annotation.
    map<int, AnnoFaceAttribute > all_anno;
    map<int, vector<int>> batchImgShape;
    if (this->output_labels_ && !has_anno_type_) {
        top_label = batch->label_.mutable_cpu_data();
    }
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        timer.Start();
        // get a anno_datum
        AnnoFaceAttributeDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
        batchImgShape[item_id].push_back (anno_datum.datum().width());
        batchImgShape[item_id].push_back (anno_datum.datum().height());
        AnnoFaceAttributeDatum distort_datum;
        AnnoFaceAttributeDatum* expand_datum = NULL;
        if (transform_param.has_distort_param()) {
            distort_datum.CopyFrom(anno_datum);
            this->data_transformer_->DistortImage(anno_datum.datum(),
                                                    distort_datum.mutable_datum());
            if (transform_param.has_expand_param()) {
                expand_datum = new AnnoFaceAttributeDatum();
                this->data_transformer_->ExpandImage(distort_datum, expand_datum);
            } else {
                expand_datum = &distort_datum;
            }
        } else {
            if (transform_param.has_expand_param()) {
                expand_datum = new AnnoFaceAttributeDatum();
                this->data_transformer_->ExpandImage(anno_datum, expand_datum);
            } else {
                expand_datum = &anno_datum;
            }
        }
        timer.Start();
        vector<int> shape =
            this->data_transformer_->InferBlobShape(expand_datum->datum());
        if (transform_param.has_resize_param()) {
            if (transform_param.resize_param().resize_mode() ==
                ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
                this->transformed_data_.Reshape(shape);
                batch->data_.Reshape(shape);
                top_data = batch->data_.mutable_cpu_data();
            } else {
                CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                    shape.begin() + 1))<<"shape: "<<shape[0]<<" "
                             <<shape[1]<<" "<<shape[2]<<" "
                             <<shape[3]<<";top_shape: "<<top_shape[0]<<" "
                             <<top_shape[1]<<" "<<top_shape[2]<<" "
                             <<top_shape[3];
            }
        } else {
            CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                shape.begin() + 1));
        }
        read_time += timer.MicroSeconds();
        // Apply data transformations (mirror, scale, crop...)
        int offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset);
        AnnoFaceAttribute transformed_anno_vec;
        if (this->output_labels_) {
            if (has_anno_type_) {
                // Transform datum and annotation_group at the same time
                this->data_transformer_->Transform(*expand_datum,
                                                &(this->transformed_data_),
                                                &transformed_anno_vec);
                all_anno[item_id] = transformed_anno_vec;
            } else {
                this->data_transformer_->Transform(expand_datum->datum(),
                                                &(this->transformed_data_));
            }
        } else {
            this->data_transformer_->Transform(expand_datum->datum(),
                                            &(this->transformed_data_));
        }
        // clear memory
        if (transform_param.has_expand_param()) {
            delete expand_datum;
        }
        trans_time += timer.MicroSeconds();
        reader_.free().push(const_cast<AnnoFaceAttributeDatum*>(&anno_datum));
    }

    // store "rich " landmark, face attributes
    if (this->output_labels_ && has_anno_type_) {
        vector<int> label_shape(4);
        if (anno_type_ == AnnoFaceAttributeDatum_AnnoType_FACEATTRIBUTE) {
            label_shape[0] = 1;
            label_shape[1] = 1;
            // Reshape the label and store the annotation.
            label_shape[2] = batch_size;
            label_shape[3] = 18;
            batch->label_.Reshape(label_shape);
            top_label = batch->label_.mutable_cpu_data();
            int idx = 0;
            for (int item_id = 0; item_id < batch_size; ++item_id) {
                AnnoFaceAttribute face = all_anno[item_id];
                top_label[idx++] = item_id;
                top_label[idx++] = face.landmark().lefteye().x();
                top_label[idx++] = face.landmark().righteye().x();
                top_label[idx++] = face.landmark().nose().x();
                top_label[idx++] = face.landmark().leftmouth().x();
                top_label[idx++] = face.landmark().rightmouth().x();
                top_label[idx++] = face.landmark().lefteye().y();
                top_label[idx++] = face.landmark().righteye().y();
                top_label[idx++] = face.landmark().nose().y();
                top_label[idx++] = face.landmark().leftmouth().y();
                top_label[idx++] = face.landmark().rightmouth().y();
                top_label[idx++] = face.faceoritation().yaw();
                top_label[idx++] = face.faceoritation().pitch();
                top_label[idx++] = face.faceoritation().roll();
                top_label[idx++] = face.gender();
                top_label[idx++] = face.glass();
                top_label[idx++] = batchImgShape[item_id][0];
                top_label[idx++] = batchImgShape[item_id][1];
            }
        } else {
            LOG(FATAL) << "Unknown annotation type.";
        }
    }
    #if 0
    for(int ii =0; ii < batch_size; ii ++){
        int idx = ii * 18;
        LOG(INFO)<<top_label[idx+1] << " " << top_label[idx+2] << " " << top_label[idx+3] << " " << top_label[idx+4] << 
        " " << top_label[idx+5] << " " << top_label[idx+6] << " " << top_label[idx+7] << " " << top_label[idx+8] << 
        " " << top_label[idx+9] << " " << top_label[idx+11] << " " << top_label[idx+12] <<
        " " << top_label[idx+13] <<  " " << top_label[idx+14] << " " << top_label[idx+15] << " " << top_label[idx+16] << 
        " " << top_label[idx+17];
    }
    #endif
    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(faceAttributeDataLayer);
REGISTER_LAYER_CLASS(faceAttributeData);

} //namespace caffe