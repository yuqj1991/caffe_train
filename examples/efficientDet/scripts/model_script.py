# this scripts is to generate caffe formate efficientDet prototxt
# -*- coding: UTF-8 -*-
import sys, os
CAFFE_ROOT = '../../../../caffe_train/'
sys.path.insert(0, CAFFE_ROOT + 'python')
import numpy as numpy
import caffe
from caffe.model_libs import *
import shutil, subprocess
batch_sampler = [
    {
        'sampler': {
        },
        'max_trials': 1,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.1,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.3,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.5,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.7,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.9,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'max_jaccard_overlap': 1.0,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
]
train_transform_param = {
    'mirror': True,
    'mean_value': [104, 117, 123],
    'resize_param': {
        'prob': 1,
        'resize_mode': P.Resize.WARP,
        'height': resize_height,
        'width': resize_width,
        'interp_mode': [
            P.Resize.LINEAR,
            P.Resize.AREA,
            P.Resize.NEAREST,
            P.Resize.CUBIC,
            P.Resize.LANCZOS4,
        ],
    },
    'distort_param': {
        'brightness_prob': 0.5,
        'brightness_delta': 32,
        'contrast_prob': 0.5,
        'contrast_lower': 0.5,
        'contrast_upper': 1.5,
        'hue_prob': 0.5,
        'hue_delta': 18,
        'saturation_prob': 0.5,
        'saturation_lower': 0.5,
        'saturation_upper': 1.5,
        'random_order_prob': 0.0,
    },
    'expand_param': {
            'prob': 0.5,
            'max_expand_ratio': 4.0,
    },
    'emit_constraint': {
        'emit_type': caffe_pb2.EmitConstraint.CENTER,
    }
}
test_transform_param = {
    'mean_value': [104, 117, 123],
    'resize_param': {
            'prob': 1,
            'resize_mode': P.Resize.WARP,
            'height': resize_height,
            'width': resize_width,
            'interp_mode': [P.Resize.LINEAR],
    },
}
saveDirPath = ""
model_name = ""
train_NetFile = "{}/{}_train.prototxt".format(saveDirPath, model_name)
test_NetFile = "{}/{}_test.prototxt".format(saveDirPath, model_name)
deploy_NetFile = "{}/{}_deploy.prototxt".format(saveDirPath, model_name)
solver_NetFile = "{}/{}_solver.prototxt".format(saveDirPath, model_name)
# labelMap 文件
label_map_file = "data/VOC0712/labelmap_voc.prototxt"

trainDataPath = ""
testDataPath = ""
resized_width = 640
resized_height = 640
resize = "{}x{}".format(resized_width, resized_height)

class caffeNet(object):
    def __init__(self, inputs, ):

    def Conv_BN_Layer():

    def Pooling_Layer():

    def Activation_Layer():

    def Concate_Layer():

    def Resnet_Block_Bottle():


def parse_prototxt_file(proto_file):

def write_prototxt_file(NetFile, model_name, model, protoDir):
    with open(NetFile, 'w') as f:
        print('name: "{}_train"'.format(model_name), file=f)
        print(model.to_proto(), file=f)
    shutil.copy(NetFile, protoDir)

def write_solver_file():