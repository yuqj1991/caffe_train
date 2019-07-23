# -*- coding:UTF-8 -*-
from __future__ import division
import numpy as np
import argparse
import sys,os
import math
import cv2
caffe_root = '../../../caffe_train/'
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
test_dir = "../../../dataset/facedata/umdfaces/JPEGImages/AFLW"
label_dir = "../../../dataset/facedata/umdfaces/label/"
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
    sum_nmse = [0.0,0.0,0.0]
    with open(labelpath, 'r') as labelfile_:
        while True:
            labelInfo = labelfile_.readline().split(' ')
            if len(labelInfo) <= 2:
                break
            x1 = float(labelInfo[0])
            x2 = float(labelInfo[1])
            x3 = float(labelInfo[2])
            x = [x1, x2, x3]
            y = [yaw, pitch, roll]
            for ii in range(3):
                sum_n = float(abs(x[ii]-y[ii]))
                sum_nmse[ii] = sum_n
    labelfile_.close()
    return sum_nmse


eval_angle = []
sum_angle= [0.0,0.0,0.0]
with open(val_list_file, 'r') as listfile_:
	while True:
		val_image = listfile_.readline()
		if not val_image:
			break
		val_imageinfo = listfile_.readline().split(' ')
		sum = detect(val_imageinfo[0], val_imageinfo[1].replace('\n', ''))
		eval_angle.append(sum)
		num_image += 1


# static
for nn in range(len(eval_angle)):
    for ii in range(3):
        sum_angle[ii] += float(eval_angle[nn][ii]/num_image)
print ("yaw : %f, pitch: %f, roll : %f," % (sum_angle[0], sum_angle[1], sum_angle[2]))


