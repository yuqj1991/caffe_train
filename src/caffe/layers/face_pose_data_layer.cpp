#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/face_pose_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe{

template <typename Dtype>
facePoseDataLayer<Dtype>::facePoseDataLayer(const LayerParameter & param)
    : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param){
}

template <typename Dtype>
facePoseDataLayer<Dtype>::~facePoseDataLayer(){
    this->StopInternalThread();
}

template <typename Dtype>
void facePoseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
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
            if (anno_type_ == AnnoFaceAttributeDatum_AnnoType_FACEPOSE) {
            label_shape[0] = 1;
            label_shape[1] = 1;
            // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
            // cpu_data and gpu_data for consistent prefetch thread. Thus we make
            // sure there is at least one bbox.
            label_shape[2] = batch_size;
            label_shape[3] = 46;
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
void facePoseDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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

    if (this->output_labels_ && !has_anno_type_) {
        top_label = batch->label_.mutable_cpu_data();
    }
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        timer.Start();
        // get a anno_datum
        AnnoFaceAttributeDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
        #if 0
        LOG(INFO) <<" image->height: "<<anno_datum.datum().height()<<" image->width: "<<anno_datum.datum().width()
                  <<" point_1: "<<anno_datum.facepose().landMark().point_1().x()<<" "<<anno_datum.facepose().landMark().point_1().y()
                  <<" point_2: "<<anno_datum.facepose().landMark().point_2().x()<<" "<<anno_datum.facepose().landMark().point_2().y()
                  <<" point_3: "<<anno_datum.facepose().landMark().point_3().x()<<" "<<anno_datum.facepose().landMark().point_3().y()
                  <<" point_4: "<<anno_datum.facepose().landMark().point_4().x()<<" "<<anno_datum.facepose().landMark().point_4().y()
                  <<" point_5: "<<anno_datum.facepose().landMark().point_5().x()<<" "<<anno_datum.facepose().landMark().point_5().y()
                  <<" point_6: "<<anno_datum.facepose().landMark().point_6().x()<<" "<<anno_datum.facepose().landMark().point_6().y()
                  <<" point_7: "<<anno_datum.facepose().landMark().point_7().x()<<" "<<anno_datum.facepose().landMark().point_7().y()
                  <<" point_8: "<<anno_datum.facepose().landMark().point_8().x()<<" "<<anno_datum.facepose().landMark().point_8().y()
                  <<" point_9: "<<anno_datum.facepose().landMark().point_9().x()<<" "<<anno_datum.facepose().landMark().point_9().y()
                  <<" point_10: "<<anno_datum.facepose().landMark().point_10().x()<<" "<<anno_datum.facepose().landMark().point_10().y()
                  <<" point_11: "<<anno_datum.facepose().landMark().point_11().x()<<" "<<anno_datum.facepose().landMark().point_11().y()
                  <<" point_12: "<<anno_datum.facepose().landMark().point_12().x()<<" "<<anno_datum.facepose().landMark().point_12().y()
                  <<" point_13: "<<anno_datum.facepose().landMark().point_13().x()<<" "<<anno_datum.facepose().landMark().point_13().y()
                  <<" point_14: "<<anno_datum.facepose().landMark().point_14().x()<<" "<<anno_datum.facepose().landMark().point_14().y()
                  <<" point_15: "<<anno_datum.facepose().landMark().point_15().x()<<" "<<anno_datum.facepose().landMark().point_15().y()
                  <<" point_16: "<<anno_datum.facepose().landMark().point_16().x()<<" "<<anno_datum.facepose().landMark().point_16().y()
                  <<" point_17: "<<anno_datum.facepose().landMark().point_17().x()<<" "<<anno_datum.facepose().landMark().point_17().y()
                  <<" point_18: "<<anno_datum.facepose().landMark().point_18().x()<<" "<<anno_datum.facepose().landMark().point_18().y()
                  <<" point_19: "<<anno_datum.facepose().landMark().point_19().x()<<" "<<anno_datum.facepose().landMark().point_19().y()
                  <<" point_20: "<<anno_datum.facepose().landMark().point_20().x()<<" "<<anno_datum.facepose().landMark().point_20().y()
                  <<" point_21: "<<anno_datum.facepose().landMark().point_21().x()<<" "<<anno_datum.facepose().landMark().point_21().y();
                  << " yaw : "<<anno_datum.facepose().faceoritation().yaw() << " "<<" pitch: "<< anno_datum.facepose().faceoritation().pitch()
                  <<" "<<" roll: "<<anno_datum.facepose().faceoritation().roll();
        LOG(INFO)<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~End Read Raw Annodation Data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
        #endif
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
        if (anno_type_ == AnnoFaceAttributeDatum_AnnoType_FACEPOSE) {
            label_shape[0] = 1;
            label_shape[1] = 1;
            // Reshape the label and store the annotation.
            label_shape[2] = batch_size;
            label_shape[3] = 46;
            batch->label_.Reshape(label_shape);
            top_label = batch->label_.mutable_cpu_data();
            int idx = 0;
            for (int item_id = 0; item_id < batch_size; ++item_id) {
                AnnoFaceAttribute face = all_anno[item_id];
                top_label[idx++] = item_id;
                top_label[idx++] = face.landMark().point_1().x();
                top_label[idx++] = face.landMark().point_2().x();
                top_label[idx++] = face.landMark().point_3().x();
                top_label[idx++] = face.landMark().point_4().x();
                top_label[idx++] = face.landMark().point_5().x();
                top_label[idx++] = face.landMark().point_6().x();
                top_label[idx++] = face.landMark().point_7().x();
                top_label[idx++] = face.landMark().point_8().x();
                top_label[idx++] = face.landMark().point_9().x();
                top_label[idx++] = face.landMark().point_10().x();
                top_label[idx++] = face.landMark().point_11().x();
                top_label[idx++] = face.landMark().point_12().x();
                top_label[idx++] = face.landMark().point_13().x();
                top_label[idx++] = face.landMark().point_14().x();
                top_label[idx++] = face.landMark().point_15().x();
                top_label[idx++] = face.landMark().point_16().x();
                top_label[idx++] = face.landMark().point_17().x();
                top_label[idx++] = face.landMark().point_18().x();
                top_label[idx++] = face.landMark().point_19().x();
                top_label[idx++] = face.landMark().point_20().x();
                top_label[idx++] = face.landMark().point_21().x();
                top_label[idx++] = face.landMark().point_1().y();
                top_label[idx++] = face.landMark().point_2().y();
                top_label[idx++] = face.landMark().point_3().y();
                top_label[idx++] = face.landMark().point_4().y();
                top_label[idx++] = face.landMark().point_5().y();
                top_label[idx++] = face.landMark().point_6().y();
                top_label[idx++] = face.landMark().point_7().y();
                top_label[idx++] = face.landMark().point_8().y();
                top_label[idx++] = face.landMark().point_9().y();
                top_label[idx++] = face.landMark().point_10().y();
                top_label[idx++] = face.landMark().point_11().y();
                top_label[idx++] = face.landMark().point_12().y();
                top_label[idx++] = face.landMark().point_13().y();
                top_label[idx++] = face.landMark().point_14().y();
                top_label[idx++] = face.landMark().point_15().y();
                top_label[idx++] = face.landMark().point_16().y();
                top_label[idx++] = face.landMark().point_17().y();
                top_label[idx++] = face.landMark().point_18().y();
                top_label[idx++] = face.landMark().point_19().y();
                top_label[idx++] = face.landMark().point_20().y();
                top_label[idx++] = face.landMark().point_21().y();
                top_label[idx++] = face.faceoritation().yaw();
                top_label[idx++] = face.faceoritation().pitch();
                top_label[idx++] = face.faceoritation().roll();
                #if 0
                LOG(INFO)<<" label point: "<<top_label[item_id*46]<<" "<<top_label[item_id*46+1]
                            <<" "<<top_label[item_id*46+2]
                            <<" "<<top_label[item_id*46+3]<<" "<<top_label[item_id*46+4]
                            <<" "<<top_label[item_id*46+5]<<" "<<top_label[item_id*46+6]
                            <<" "<<top_label[item_id*46+7]<<" "<<top_label[item_id*46+8]
                            <<" "<<top_label[item_id*46+9]<<" "<<top_label[item_id*46+10]
                            <<" "<<top_label[item_id*46+11]<<" "<<top_label[item_id*46+12]
                            <<" "<<top_label[item_id*46+13]<<" "
                            <<" "<<top_label[item_id*46+14]<<" "<<top_label[item_id*46+15]
                            <<" "<<top_label[item_id*46+16]<<" "<<top_label[item_id*46+17]
                            <<" "<<top_label[item_id*46+18]<<" "<<top_label[item_id*46+19]
                            <<" "<<top_label[item_id*46+20]<<" "<<top_label[item_id*46+21]
                            <<" "<<top_label[item_id*46+22]<<" "<<top_label[item_id*46+23]
                            <<" "<<top_label[item_id*46+24]<<" "<<top_label[item_id*46+25]
                            <<" "<<top_label[item_id*46+26]<<" "<<top_label[item_id*46+27]
                            <<" "<<top_label[item_id*46+28]<<" "<<top_label[item_id*46+29]
                            <<" "<<top_label[item_id*46+30]<<" "<<top_label[item_id*46+32]
                            <<" "<<top_label[item_id*46+32]<<" "<<top_label[item_id*46+33]
                            <<" "<<top_label[item_id*46+34]<<" "<<top_label[item_id*46+35]
                            <<" "<<top_label[item_id*46+36]<<" "<<top_label[item_id*46+37]
                            <<" "<<top_label[item_id*46+38]<<" "<<top_label[item_id*46+39]
                            <<" "<<top_label[item_id*46+40]<<" "<<top_label[item_id*46+41]
                            <<" "<<top_label[item_id*46+42]<<" "<<top_label[item_id*46+43]
                            <<" "<<top_label[item_id*46+44]<<" "<<top_label[item_id*46+45];
                #endif
            }
        } else {
            LOG(FATAL) << "Unknown annotation type.";
        }
    }
    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(facePoseDataLayer);
REGISTER_LAYER_CLASS(facePoseData);

} //namespace caffe