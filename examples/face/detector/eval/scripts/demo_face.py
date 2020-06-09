import numpy as np
import argparse
import sys,os  
import cv2
import time
caffe_root = '../../../../../../caffe_train/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

minMargin = 36

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='.prototxt file for inference', default = '../net/face_detector.prototxt')
    parser.add_argument('--weights', type=str, help='.caffemodel file for inference', default = '../net/face_detector.caffemodel')
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
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(net_file,caffe_model,caffe.TEST) 

CLASSES = ('background', 'face')
gender_content = ('male', 'female')
glasses_content = ('wearing glasses', 'not wearing glasses')


test_dir = "../images"

def max_(m,n):
	if m > n:
		return m
	return n


def min_(m,n):
	if m > n:
		return n
	return m


def preprocessdet(src, size):
    img = cv2.resize(src, size)
    img = img - [103.94, 116.78, 123.68]
    img = img * 0.007843
    return img


def preprocess(src, size):
    img = cv2.resize(src, size)
    img = img - 127.5
    img = img * 0.007843
    return img


def post_facedet(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return box.astype(np.int32), conf, cls


def post_faceattributes(img, out):
    h = img.shape[0]
    w = img.shape[1]
    facepoints = out['multiface_output'][0,0:10] * np.array([w, w, w, w, w, h, h, h, h, h])
    faceangle = out['multiface_output'][0,10:13]
    gender = out['multiface_output'][0,13:15]
    gender_index = np.argmax(gender)
    glass = out['multiface_output'][0,15:17]
    glass_index = np.argmax(glass)
    return facepoints.astype(np.int32), faceangle, gender_content[gender_index], glasses_content[glass_index]


def detect(imgfile):
   frame = cv2.imread(imgfile)
   h = frame.shape[0]
   w = frame.shape[1]
   inputSize = (net.blobs['data'].data.shape[3], net.blobs['data'].data.shape[2])
   img = preprocessdet(frame, inputSize)
   img = img.astype(np.float32)
   
   img = img.transpose((2, 0, 1))
   net.blobs['data'].data[...] = img
   net.forward()
   print(net.blobs['Det_1x1_out_24_0'].data.shape)
   print(net.blobs['Det_1x1_out_24_0'].data)
   #print(net.forward())

if __name__=="__main__":
    imgfile = "../images/grace_hopper.jpg"
    detect(imgfile)
