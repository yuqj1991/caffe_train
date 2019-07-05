#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/faceData_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe{

template <typename Dtype>
faceAnnoDataLayer<Dtype>::faceAnnoDataLayer(const LayerParameter & param)
    : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param){
}

template <typename Dtype>
faceAnnoDataLayer<Dtype>::~faceAnnoDataLayer(){
    this->StopInternalThread();
}

template <typename Dtype>
void faceAnnoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
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
    AnnoFaceDatum& anno_datum = *(reader_.full().peek());
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
            if (anno_type_ == AnnoFaceDatum_AnnotationType_FACEMARK) {
            // Since the number of lanmarks of one person can be constant for each image,
            // and another point is each image only have one person face, so we store the lable 
            // in a specific formate. In specific:
            // All landmarks  are stored in one spatial plane (# x1,...,x5; #y1,...,y5),which has five point 
            // and they are left eye, right eye, nose , left mouse endpoint, right mouse endpoint. And the labels formate:
            // [item_id(image_id), x1,..,x5, y1,...,y5, gender, glasses]
            label_shape[0] = 1;
            label_shape[1] = 1;
            // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
            // cpu_data and gpu_data for consistent prefetch thread. Thus we make
            // sure there is at least one bbox.
            label_shape[2] = batch_size;
            label_shape[3] = 13;
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
void faceAnnoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
    AnnoFaceDatum& anno_datum = *(reader_.full().peek());
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(anno_datum.datum());
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);

    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

      // Store transformed annotation.
    map<int, AnnotationFace > all_anno;

    if (this->output_labels_ && !has_anno_type_) {
        top_label = batch->label_.mutable_cpu_data();
    }
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        timer.Start();
        // get a anno_datum
        AnnoFaceDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
    #if 0
        LOG(INFO)<<" START READ RAW ANNODATUM=================================================";
        LOG(INFO) <<" image->height: "<<anno_datum.datum().height()<<" image->width: "<<anno_datum.datum().width()
                  <<" point_1: "<<anno_datum.annoface().markface().x1()<<" "<<anno_datum.annoface().markface().y1()
                  <<" point_2: "<<anno_datum.annoface().markface().x2()<<" "<<anno_datum.annoface().markface().y2()
                  <<" point_3: "<<anno_datum.annoface().markface().x3()<<" "<<anno_datum.annoface().markface().y3()
                  <<" point_4: "<<anno_datum.annoface().markface().x4()<<" "<<anno_datum.annoface().markface().y4()
                  <<" point_5: "<<anno_datum.annoface().markface().x5()<<" "<<anno_datum.annoface().markface().y5();
        LOG(INFO)<<" END READ RAW ANNODATUM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
    #endif
        AnnoFaceDatum distort_datum;
        AnnoFaceDatum* rotate_datum = NULL;
        AnnoFaceDatum* expand_datum = NULL;
        if (transform_param.has_distort_param()) {
            distort_datum.CopyFrom(anno_datum);
            this->data_transformer_->DistortImage(anno_datum.datum(),
                                                    distort_datum.mutable_datum());
            if(transform_param.has_rotate_param()){
                rotate_datum = new AnnoFaceDatum();
                this->data_transformer_->RotateImage(distort_datum, rotate_datum);
                if (transform_param.has_expand_param()) {
                    expand_datum = new AnnoFaceDatum();
                    this->data_transformer_->ExpandImage(*rotate_datum, expand_datum);
                }else{
                    expand_datum = rotate_datum;
                }
            }else{
                if (transform_param.has_expand_param()) {
                    expand_datum = new AnnoFaceDatum();
                    this->data_transformer_->ExpandImage(distort_datum, expand_datum);
                }else{
                    expand_datum = &distort_datum;
                }
            }
        } else {
            if(transform_param.has_rotate_param()){
                rotate_datum = new AnnoFaceDatum();
                this->data_transformer_->RotateImage(anno_datum, rotate_datum);
                if (transform_param.has_expand_param()) {
                    expand_datum = new AnnoFaceDatum();
                    this->data_transformer_->ExpandImage(*rotate_datum, expand_datum);
                }else{
                    expand_datum = rotate_datum;
                }
            }else{
                if (transform_param.has_expand_param()) {
                    expand_datum = new AnnoFaceDatum();
                    this->data_transformer_->ExpandImage(anno_datum, expand_datum);
                }else{
                    expand_datum = &anno_datum;
                }
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
                    shape.begin() + 1))<<" as follows, shape: "<<shape[0]<<" "
                             <<shape[1]<<" "<<shape[2]<<" "
                             <<shape[3]<<"; top_shape: "<<top_shape[0]<<" "
                             <<top_shape[1]<<" "<<top_shape[2]<<" "
                             <<top_shape[3]<<" item_id: "<<item_id;
            }
        } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                shape.begin() + 1));
        }
        read_time += timer.MicroSeconds();
        // Apply data transformations (mirror, scale, crop...)
        int offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset);
        AnnotationFace transformed_anno_vec;
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
        reader_.free().push(const_cast<AnnoFaceDatum*>(&anno_datum));
    }

    // store "rich " landmark, face attributes
    if (this->output_labels_ && has_anno_type_) {
        vector<int> label_shape(4);
        if (anno_type_ == AnnoFaceDatum_AnnotationType_FACEMARK) {
            label_shape[0] = 1;
            label_shape[1] = 1;
            // Reshape the label and store the annotation.
            label_shape[2] = batch_size;
            label_shape[3] = 13;
            batch->label_.Reshape(label_shape);
            top_label = batch->label_.mutable_cpu_data();
            int idx = 0;
            for (int item_id = 0; item_id < batch_size; ++item_id) {
                AnnotationFace face = all_anno[item_id];
                top_label[idx++] = item_id;
                top_label[idx++] = face.markface().x1();
                top_label[idx++] = face.markface().x2();
                top_label[idx++] = face.markface().x3();
                top_label[idx++] = face.markface().x4();
                top_label[idx++] = face.markface().x5();
                top_label[idx++] = face.markface().y1();
                top_label[idx++] = face.markface().y2();
                top_label[idx++] = face.markface().y3();
                top_label[idx++] = face.markface().y4();
                top_label[idx++] = face.markface().y5();
                top_label[idx++] = face.gender();
                top_label[idx++] = face.glasses();
                #if 0
                LOG(INFO)<<" label point: "<<top_label[item_id*13]<<" "<<top_label[item_id*13+1]
                            <<" "<<top_label[item_id*13+2]
                            <<" "<<top_label[item_id*13+3]<<" "<<top_label[item_id*13+3]
                            <<" "<<top_label[item_id*13+5]<<" "<<top_label[item_id*13+6]
                            <<" "<<top_label[item_id*13+7]<<" "<<top_label[item_id*13+8]
                            <<" "<<top_label[item_id*13+9]<<" "<<top_label[item_id*13+10]
                            <<" "<<top_label[item_id*13+11]<<" "<<top_label[item_id*13+12]
                            <<" "<<top_label[item_id*13+13];
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

INSTANTIATE_CLASS(faceAnnoDataLayer);
REGISTER_LAYER_CLASS(faceAnnoData);

} //namespace caffe