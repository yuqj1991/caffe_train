#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP__

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe{

template <typename Dtype>
class faceAnnoDataLayer: public BasePrefetchingDataLayer<Dtype>{
    public:
        explicit faceAnnoDataLayer(const LayerParameter & param);
        virtual ~faceAnnoDataLayer();
        virtual void layerDataSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        // faceDataLayer uses DataReader instead for sharing for parallelism
        virtual inline bool ShareInParallel() const { return false; }
        virtual inline const char* type() const { return "faceAnnoData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int MinTopBlobs() const { return 1; }

    protected:
        virtual void load_batch(Batch<Dtype>* batch);

        DataReader<AnnoFaceDatum> reader_;
        bool has_anno_type_;
        AnnoFaceDatum_AnnotationType anno_type_;
};

}  //cafe

#endif // CAFFE_DATA_LAYER_HPP_