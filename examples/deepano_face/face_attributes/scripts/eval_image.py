# -*- coding:UTF-8 -*-
from __future__ import division
import numpy as np
import argparse
import sys,os  
import cv2
caffe_root = '../../../../../face_train/'
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
test_dir = "../../../../../dataset/facedata/mtfl/JPEGImages/AFLW"
label_dir = "../../../../../dataset/facedata/mtfl/label/"
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

gender_content = ('male', 'female')
glasses_content = ('wearing glasses', 'not wearing glasses')
headpose_content = ('left profile', 'left', 'frontal', 'right', 'right profile')


def preprocess(src):
    img = cv2.resize(src, (96,96))
    img = img - 127.5
    img = img * 0.007843
    return img


def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    facepoints = out['multiface_output'][0, 0:10] * np.array([w, w, w, w, w, h, h, h, h, h])
    gender = out['multiface_output'][0, 10:12]
    gender_index = np.argmax(gender)
    glasses = out['multiface_output'][0, 12:14]
    glasses_index = np.argmax(glasses)
    headpose = out['multiface_output'][0, 14:19]
    headpose_index = np.argmax(headpose)
    return (facepoints.astype(np.int32), gender_index, glasses_index,
           headpose_index)


def detect(imgfile):
    origimg = cv2.imread(imgfile)
    w = origimg.shape[1]
    img = preprocess(origimg)
    labelpath = label_dir+imgfile.split('/')[-1].split('.jpg')[0]
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    box, gender, glasses, headpose = postprocess(origimg, out)
    sum_nmse = [0.0,0.0,0.0,0.0,0.0]

    num_correct_gender = 0
    num_correct_glasses = 0
    num_correct_headpose = 0

    with open(labelpath, 'r') as labelfile_:
        while True:
            labelInfo = labelfile_.readline().split(' ')
            if len(labelInfo) <= 2:
                break
            x1 = float(labelInfo[0])
            x2 = float(labelInfo[1])
            x3 = float(labelInfo[2])
            x4 = float(labelInfo[3])
            x5 = float(labelInfo[4])
            y1 = float(labelInfo[5])
            y2 = float(labelInfo[6])
            y3 = float(labelInfo[7])
            y4 = float(labelInfo[8])
            y5 = float(labelInfo[9])
            gender_gt = int(labelInfo[10]) - 1
            glass_gt = int(labelInfo[11]) - 1
            headpose_gt = int(labelInfo[12]) - 1
            if gender == gender_gt:
                num_correct_gender = 1
            if glasses == glass_gt:
                num_correct_glasses = 1
            if headpose == headpose_gt:
                num_correct_headpose = 1
            x = [x1, x2, x3, x4, x5]
            y = [y1, y2, y3, y4, y5]
            for ii in range(5):
                sum_nmse[ii] = pow((x[ii]-box[ii]), 2) + pow((y[ii]-box[ii+5]), 2)
                sum_nmse[ii] = float(pow(sum_nmse[ii], 0.5))/w
    labelfile_.close()
    return sum_nmse, num_correct_gender, num_correct_glasses, num_correct_headpose


mtfl_eval_landmakrs = []
mtfl_eval_gender = 0
mtfl_eval_glass =0
mtfl_eval_headpose = 0
sum_landmark = [0.0,0.0,0.0,0.0,0.0]
for f in os.listdir(test_dir):
    sum, num_gender, num_glasses, num_headpose = detect(test_dir + "/" + f)
    mtfl_eval_landmakrs.append(sum)
    mtfl_eval_gender += num_gender
    mtfl_eval_glass += num_glasses
    mtfl_eval_headpose +=num_headpose
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


