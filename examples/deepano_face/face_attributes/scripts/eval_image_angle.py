# -*- coding:UTF-8 -*-
from __future__ import division
import numpy as np
import argparse
import sys,os
import math
import cv2
caffe_root = '../../../../../caffe_deeplearning_train/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type = str, required=True, help='.caffemodel file for inference')
    return parser


parser1 = make_parser()
args = parser1.parse_args()
net_file= args.model
caffe_model= args.weights
test_dir = "../../../../../dataset/facedata/umdfaces/JPEGImages/AFLW"
label_dir = "../../../../../dataset/facedata/umdfaces/label/"
val_list_file = 'testing.txt'
num_image = 0

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(net_file,caffe_model, caffe.TEST)


def preprocess(src):
    img = cv2.resize(src, (48, 48))
    img = img - 127.5
    img = img * 0.007843
    return img

def detect(imgfile, labelpath):
    origimg = cv2.imread(imgfile)
    w = origimg.shape[1]
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    yaw, pitch, roll = out['conv6_angle'][0, 0:3] * 360
    sum_nmse = [0.0,0.0,0.0,0.0,0.0]

    correct_gender = False
    correct_glasses = False
    correct_headpose = False

    with open(labelpath, 'r') as labelfile_:
        while True:
            labelInfo = labelfile_.readline().split(' ')
            if len(labelInfo) <= 2:
                break
            x1 = float(labelInfo[0])
            x2 = float(labelInfo[1])
            x3 = float(labelInfo[2])

            intal_eye_w = float(pow((pow((x1-x2), 2) + pow((y1-y2), 2)), 0.5))
            for ii in range(5):
                sum_n = float(pow(pow((x[ii]-point[ii]), 2) + pow((y[ii]-point[ii+5]), 2), 0.5))
                sum_nmse[ii] = sum_n/intal_eye_w
    labelfile_.close()
    return sum_nmse, correct_gender, correct_glasses, correct_headpose


mtfl_eval_landmakrs = []
mtfl_eval_gender = 0
mtfl_eval_glass =0
mtfl_eval_headpose = 0
sum_landmark = [0.0,0.0,0.0,0.0,0.0]
with open(val_list_file, 'r') as listfile_:
	while True:
		val_image = listfile_.readline()
		if not val_image:
			break
		val_imageinfo = listfile_.readline().split(' ')
		sum, gender, glasses, headpose = detect(val_imageinfo[0], val_imageinfo[1].replace('\n', ''))
		mtfl_eval_landmakrs.append(sum)
		if gender:
			mtfl_eval_gender += 1
		if glasses:
			mtfl_eval_glass += 1
		if headpose:
			mtfl_eval_headpose += 1
		num_image += 1

# static
for nn in range(len(mtfl_eval_landmakrs)):
    for ii in range(5):
        sum_landmark[ii] += float(mtfl_eval_landmakrs[nn][ii]/num_image)
eval_gender = float(mtfl_eval_gender/num_image)
eval_glass = float(mtfl_eval_glass/num_image)
eval_headpose = float(mtfl_eval_headpose/num_image)

print ("eval_gender: %f, eval_glass: %f, eval_headpose: %f" % (eval_gender, eval_glass, eval_headpose))
print ("left eye : %f, right eye: %f, nose : %f, left cornor mouth: %f, right cornor mouth: %f" % (sum_landmark[0],
                                                                                                   sum_landmark[1],
                                                                                                   sum_landmark[2],
                                                                                                   sum_landmark[3],
                                                                                                   sum_landmark[4]))


