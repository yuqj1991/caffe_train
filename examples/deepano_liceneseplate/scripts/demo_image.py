# -*- coding:UTF-8 -*-
import numpy as np
import argparse
import sys,os  
import cv2
caffe_root = '../../../../caffe_deeplearning_train/'
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
    return parser
parser1 = make_parser()
args = parser1.parse_args()
net_file= args.model
caffe_model= args.weights
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

CLASSES = ('background',
           'liceneseplate')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    chin = out['detection_out'][0,0,:,7]
    eng = out['detection_out'][0,0,:,8]
    letter_1 = out['detection_out'][0,0,:,9]
    letter_2 = out['detection_out'][0,0,:,10]
    letter_3 = out['detection_out'][0,0,:,11]
    letter_4 = out['detection_out'][0,0,:,12]
    letter_5 = out['detection_out'][0,0,:,13]
    print(chin)
    print(eng)
    print(letter_1)
    print(letter_2)
    print(letter_3)
    print(letter_4)
    print(letter_5)
    return (box.astype(np.int32), conf, cls, chin.astype(np.int32), eng.astype(np.int32), letter_1.astype(np.int32), letter_2.astype(np.int32), letter_3.astype(np.int32), letter_4.astype(np.int32), letter_5.astype(np.int32) )

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    print("out shape",out['detection_out'].shape)  
    box, conf, cls, chin, eng, letter_1, letter_2, letter_3, letter_4, letter_5 = postprocess(origimg, out)

    for i in range(len(box)):
       if conf[i]>=0.25:
           p1 = (box[i][0], box[i][1])
           p2 = (box[i][2], box[i][3])
           cv2.rectangle(origimg, p1, p2, (0,255,0))
           p3 = (max(p1[0], 15), max(p1[1], 15))
           lpstring = provinces[chin[i]] + alphabets[eng[i]] + ads[letter_1[i]] +ads[letter_2[i]] + ads[letter_3[i]] + ads[letter_4[i]] + ads[letter_5[i]]		
           title = "%s: %s" % (CLASSES[int(cls[i])],lpstring)
           cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("facedetector", origimg)
 
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break

