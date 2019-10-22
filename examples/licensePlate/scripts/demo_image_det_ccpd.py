# -*- coding:UTF-8 -*-
import numpy as np
import argparse
import sys,os  
import cv2
from PIL import Image, ImageDraw, ImageFont
caffe_root = '../../../../caffe_train/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

provNum, alphaNum, adNum = 34, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file for inference')
    parser.add_argument('--ccpdmodel', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--ccpdweights', type=str, required=True, help='.caffemodel file for inference')
    return parser
parser1 = make_parser()
args = parser1.parse_args()
net_file= args.model
caffe_model= args.weights
ccpd_file= args.ccpdmodel
ccpd_model= args.ccpdweights
test_dir = "../images"

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(net_file,caffe_model,caffe.TEST)  
ccpd_net = caffe.Net(ccpd_file,ccpd_model,caffe.TEST)  

CLASSES = ('background',
           'liceneseplate')


def max_(m,n):
	if m > n:
		return m
	return n


def min_(m,n):
	if m > n:
		return n
	return m


font = ImageFont.truetype('NotoSansCJK-Black.ttc', 20)
fillColor = (255,0,0)

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img


def preprocessccpd(src):
    img = cv2.resize(src, (128, 64))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    h = origimg.shape[0]
    w = origimg.shape[1]
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    print("out shape",out['detection_out'].shape)  
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
       if conf[i]>=0.25:
           p1 = (box[i][0], box[i][1])
           p2 = (box[i][2], box[i][3])
           x1 = max_(0, box[i][0])
           x2 = min_(box[i][2], w)
           y1 = max_(0, box[i][1])
           y2 = min_(box[i][3], h)
           ori_img = origimg[y1:y2, x1:x2, :]
           ccpdimg = preprocessccpd(ori_img)
           ccpdimg = ccpdimg.astype(np.float32)
           ccpdimg = ccpdimg.transpose((2, 0, 1))
           ccpd_net.blobs['data'].data[...] = ccpdimg
           ccpd_out = ccpd_net.forward()
           ccpdbox = ccpd_out['ccpd_output'][0,0,:,0:7]
           ccpdstring = provinces[int(ccpdbox[0][0])] + alphabets[int(ccpdbox[0][1])] + ads[int(ccpdbox[0][2])] + ads[int(ccpdbox[0][3])] + ads[int(ccpdbox[0][4])]+ ads[int(ccpdbox[0][5])] + ads[int(ccpdbox[0][6])] 
           print(ccpdstring)
           cv2.rectangle(origimg, p1, p2, (0,255,0))
           p3 = (max(p1[0], 15), max(p1[1], 15))
           title = "%s" % (CLASSES[int(cls[i])])
           if not isinstance(title, unicode): 
               title = title.decode('utf8')
           img_PIL = Image.fromarray(cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB))
           draw = ImageDraw.Draw(img_PIL)
           draw.text(p3, title, font=font, fill=fillColor) 
           origimg = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    cv2.imshow("facedetector", origimg)
 
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break

