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
void facePoseDataLayer<Dtype>::layerDataSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
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
    AnnoFacePoseData& anno_datum = *(reader_.full().peek());

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
        vector<int> label_shape(46, 1);
        if (has_anno_type_) {
            anno_type_ = anno_datum.type();
            if (anno_type_ == AnnoFacePoseDatum_AnnoType_FACEPOSE) {
            // Since the number of lanmarks of one person can be constant for each image,
            // and another point is each image only have one person face, so we store the lable 
            // in a specific formate. In specific:
            // All landmarks  are stored in one spatial plane (# x1,...,x21; #y1,...,y21), and three head pose which has 21 point 
            // and they are left eye, right eye, nose , left mouse endpoint, right mouse endpoint. And the labels formate:
            // [item_id(image_id), x1,..,x21, y1,...,y21, yaw,  pitch, roll]
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
    AnnoFacePoseDatum& anno_datum = *(reader_.full().peek());
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(anno_datum.datum());
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);

    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

      // Store transformed annotation.
    map<int, AnnoFacePose > all_anno;

    if (this->output_labels_ && !has_anno_type_) {
        top_label = batch->label_.mutable_cpu_data();
    }
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        timer.Start();
        // get a anno_datum
        AnnoFacePoseDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
        #if 0
        int size_group = anno_datum.annotation_group_size();
        LOG(INFO)<<" START READ RAW ANNODATUM=================================================";
        for(int ii=0; ii< size_group; ii++)
        {
            const AnnotationGroup& anno_group = anno_datum.annotation_group(ii);
            int anno_size = anno_group.annotation_size();
            for(int jj=0; jj<anno_size; jj++)
            {
                const Annotation& anno = anno_group.annotation(jj);
                const NormalizedBBox& bbox = anno.bbox();
                LOG(INFO) <<" bbox->xmin: "<<bbox.xmin()<<" bbox->ymin: "<<bbox.ymin()
                        <<" bbox->xmax: "<<bbox.xmax()<<" bbox->ymax: "<<bbox.ymax()
                        <<" bbox->blur: "<<bbox.blur()<<" bbox->occlusion: "<<bbox.occlusion();
            }
        }
        LOG(INFO)<<" END READ RAW ANNODATUM+++++++++++++++++++++++++++++++++++++++++++++++++++";
    #endif
        AnnoFacePoseDatum distort_datum;
        AnnoFacePoseDatum* expand_datum = NULL;
        if (transform_param.has_distort_param()) {
            distort_datum.CopyFrom(anno_datum);
            this->data_transformer_->DistortImage(anno_datum.datum(),
                                                    distort_datum.mutable_datum());
            if (transform_param.has_expand_param()) {
                expand_datum = new AnnoFacePoseDatum();
                this->data_transformer_->ExpandImage(distort_datum, expand_datum);
            } else {
                expand_datum = &distort_datum;
            }
            } else {
            if (transform_param.has_expand_param()) {
                expand_datum = new AnnoFacePoseDatum();
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
                    shape.begin() + 1));
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
                // Otherwise, store the label from datum.
                // CHECK(expand_datum->datum().has_label()) << "Cannot find any label.";
                // top_label[item_id] = expand_datum->datum().label();
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
            label_shape[3] = 46;
            batch->label_.Reshape(label_shape);
            top_label = batch->label_.mutable_cpu_data();
            int idx = 0;
            for (int item_id = 0; item_id < batch_size; ++item_id) {
                AnnoFacePose face = all_anno[item_id];
                top_label[idx++] = item_id;
                top_label[idx++] = face.faceCour().x1();
                top_label[idx++] = face.faceCour().x2();
                top_label[idx++] = face.faceCour().x3();
                top_label[idx++] = face.faceCour().x4();
                top_label[idx++] = face.faceCour().x5();
                top_label[idx++] = face.faceCour().x6();
                top_label[idx++] = face.faceCour().x7();
                top_label[idx++] = face.faceCour().x8();
                top_label[idx++] = face.faceCour().x9();
                top_label[idx++] = face.faceCour().x10();
                top_label[idx++] = face.faceCour().x11();
                top_label[idx++] = face.faceCour().x12();
                top_label[idx++] = face.faceCour().x13();
                top_label[idx++] = face.faceCour().x14();
                top_label[idx++] = face.faceCour().x15();
                top_label[idx++] = face.faceCour().x16();
                top_label[idx++] = face.faceCour().x17();
                top_label[idx++] = face.faceCour().x18();
                top_label[idx++] = face.faceCour().x19();
                top_label[idx++] = face.faceCour().x20();
                top_label[idx++] = face.faceCour().x21();
                top_label[idx++] = face.faceCour().y1();
                top_label[idx++] = face.faceCour().y2();
                top_label[idx++] = face.faceCour().y3();
                top_label[idx++] = face.faceCour().y4();
                top_label[idx++] = face.faceCour().y5();
                top_label[idx++] = face.faceCour().y6();
                top_label[idx++] = face.faceCour().y7();
                top_label[idx++] = face.faceCour().y8();
                top_label[idx++] = face.faceCour().y9();
                top_label[idx++] = face.faceCour().y10();
                top_label[idx++] = face.faceCour().y11();
                top_label[idx++] = face.faceCour().y12();
                top_label[idx++] = face.faceCour().y13();
                top_label[idx++] = face.faceCour().y14();
                top_label[idx++] = face.faceCour().y15();
                top_label[idx++] = face.faceCour().y16();
                top_label[idx++] = face.faceCour().y17();
                top_label[idx++] = face.faceCour().y18();
                top_label[idx++] = face.faceCour().y19();
                top_label[idx++] = face.faceCour().y20();
                top_label[idx++] = face.faceCour().y21();
                top_label[idx++] = face.yaw();
                top_label[idx++] = face.pitch();
                top_label[idx++] = face.raw();
            }
        } else {
            LOG(FATAL) << "Unknown annotation type.";
        }
    }
#if 0
  LOG(INFO)<< "start printf &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& single image: num_bboxes: "<<num_bboxes;
  const Dtype* top_label_data = batch->label_.cpu_data();
  for(int ii=0; ii < num_bboxes; ii++)
  {
    int id = ii*10;
    LOG(INFO) <<"batch_id: "<<top_label_data[id]<<" anno_label: "<<top_label_data[id+1]
              <<" anno.instance_id: "<<top_label_data[id+2];
    LOG(INFO)  <<"bbox->xmin: "<<top_label_data[id+3]<<" bbox->ymin: "<<top_label_data[id+4]
              <<" bbox->xmax: "<<top_label_data[id+5]<<" bbox->ymax: "<<top_label_data[id+6]
              <<" bbox->blur: "<<top_label_data[id+7]<<" bbox->occlusion: "<<top_label_data[id+8]
              <<" bbox->difficult: "<<top_label_data[id+9];
  }
  LOG(INFO)<< "finished **************************************************** end ";
#endif 

    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(faceAnnoPoseDataLayer);
REGISTER_LAYER_CLASS(faceAnnoPoseData);

} //namespace caffe