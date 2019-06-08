import numpy as np
import argparse
import sys,os  
import cv2
caffe_root = '../../../../../caffe_deeplearning_train/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file for inference')
    return parser
parser1 = make_parser()
args = parser1.parse_args()
net_file= args.model
caffe_model= args.weights
test_dir ='../images'
#test_dir = "../../../../../dataset/facedata/mtfl/JPEGImages/AFLW"

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(net_file,caffe_model,caffe.TEST)  


def preprocess(src):
    img = cv2.resize(src, (48,48))
    img = img - 127.5
    img = img * 0.007843
    return img


def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    yaw, pitch, roll = out['conv6_angle'][0, 0:3] * 360

    title = "yaw: %f, pitch: %s,roll: %s" % (gender_content[gender], glasses_content[glasses], headpose_content[headpose])
    cv2.putText(origimg, title, (10,10), cv2.FONT_ITALIC, 0.3, (0, 255, 0), 1)
    print(title)
    cv2.imshow("faceangle", origimg)
 
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break

