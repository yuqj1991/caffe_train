from __future__ import division
import numpy as np
import argparse
import sys,os
import math
import cv2
import common
import datetime
import pandas as pd

SOURCE_IMG_FILE_FOLDER = '../../../dataset/facedata/umdface/JPEGImages/' 
ImageSetFileForder = '../../../dataset/facedata/umdface/ImageSet/Main/'
trainFileName = ['umdfaces_batch2_ultraface.csv']
testFileName = ['umdfaces_batch3_ultraface.csv']
trainSet = [common.ORI_BATCH2]
testSet = [ common.ORI_BATCH3]


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
    img = cv2.resize(src, (96,96))
    img = img - 127.5
    img = img * 0.007843
    return img


def postprocess(img, out):
    glasses = out['multiface_output'][0, 12:14]
    glasses_index = np.argmax(glasses)
    return glasses_index


def batch_work(ori, csvFile):
    for ii in range(len(ori)):
        print("start: %d", ii)
        df = common.read_from_file(SOURCE_IMG_FILE_FOLDER+ori[ii]+csvFile[ii])
        csvFile = SOURCE_IMG_FILE_FOLDER+ori[ii]+csvFile[ii].split('.csv')[0] + '_new.csv'
        glassSet = []
        for row in df.iterrows():
            #Extract Important Imformation
            file_name = row[1]['FILE']
            full_path_image_name = SOURCE_IMG_FILE_FOLDER + ori[ii] + file_name
            fullImg = os.path.abspath(full_path_image_name) + '\n'
            roi_x = int(row[1]['FACE_X'])
            roi_y = int(row[1]['FACE_Y'])
            roi_w = int(row[1]['FACE_WIDTH'])
            roi_h = int(row[1]['FACE_HEIGHT'])
            orc_img = cv2.imread(os.path.abspath(full_path_image_name))
            ori_img = orc_img[roi_y :roi_y+roi_h, roi_x:roi_x+roi_w, :]
            img = preprocess(ori_img)
            img = img.astype(np.float32)
            img = img.transpose((2, 0, 1))
            net.blobs['data'].data[...] = img
            out = net.forward()
            glasses = postprocess(ori_img, out)
            print(glasses)
            glassSet.append(glasses)
        df['BOOLGLASS'] = glassSet
        df.to_csv(csvFile, mode ='a', index = False)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    trainset = ImageSetFileForder + 'training_umdface_pose.txt'
    testset = ImageSetFileForder + 'testing_umdface_pose.txt'
    batch_work(trainSet, trainFileName)
    batch_work(testSet, testFileName)
    end_time = datetime.datetime.now()
