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


chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ]

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ssd_model_def', default= '{}examples/licensePlate/net/SSD_300X300/mssd512_voc.prototxt'.format(caffe_root))
	parser.add_argument('--ssd_image_resize', default=300, type=int)
	parser.add_argument('--ssd_model_weights', default= '{}examples/licensePlate/net/SSD_300X300/mssd512_voc.caffemodel'.format(caffe_root))
	parser.add_argument('--recog_model_def', default='{}examples/licensePlate/net/LPR/deploy.prototxt'.format(caffe_root))
	parser.add_argument('--recog_image_width', default=128, type=int)
	parser.add_argument('--recog_image_height', default=32, type=int)
	parser.add_argument('--recog_model_weights', default='{}examples/licensePlate/net/LPR/lpr_recognition.caffemodel'.format(caffe_root))
	return parser


parser1 = make_parser()
args = parser1.parse_args()
net_file= args.ssd_model_def
caffe_model= args.ssd_model_weights
ccpd_file= args.recog_model_def
ccpd_model= args.recog_model_weights
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

inputShape = net.blobs['data'].data.shape
det_inputSize = (inputShape[3], inputShape[2])

inputShape = ccpd_net.blobs['data'].data.shape
rec_inputSize = (inputShape[3], inputShape[2])

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


def net_result_to_string(res):
	str = ''
	for x in res[0]:
		index = int(x[0][0])
		#print('idx:', index)
		if index != -1:
			str += chars[index]
	return str



font = ImageFont.truetype('NotoSansCJK-Black.ttc', 20)
fillColor = (255,0,0)
pixel_means= [103.53, 116.28, 123.675]
pixel_stds= [0.0174, 0.0175, 0.0171]

means= [127.5, 127.5, 127.5]
stds= [0.007843, 0.007843, 0.007843]
def preprocess(src, imgSize, means, stds):
    img = cv2.resize(src, imgSize)
    img = img - means
    img = img * stds
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
    img = preprocess(origimg, det_inputSize, pixel_means, pixel_stds)
    
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
           ccpdimg = preprocess(ori_img, rec_inputSize, means, stds)
           ccpdimg = ccpdimg.astype(np.float32)
           ccpdimg = ccpdimg.transpose((2, 0, 1))
           ccpd_net.blobs['data'].data[...] = ccpdimg
           #ccpd_out = ccpd_net.forward()
           #ccpdbox = ccpd_out['result'][0,0,:,0:7]
           #ccpdstring = provinces[int(ccpdbox[0][0])] + alphabets[int(ccpdbox[0][1])] + ads[int(ccpdbox[0][2])] + ads[int(ccpdbox[0][3])] + ads[int(ccpdbox[0][4])]+ ads[int(ccpdbox[0][5])] + ads[int(ccpdbox[0][6])] 
           result = ccpd_net.forward()['result']
           ccpdstring = net_result_to_string(result)
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

