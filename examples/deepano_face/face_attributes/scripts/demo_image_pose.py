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
#test_dir = "../umdfaceimg"
test_dir ='../images'


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
    img = cv2.resize(src, (99,99))
    img = img - 127.5
    img = img * 0.007843
    return img


def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    print("h: %d, w: %d"%(h, w))
    facepoints = out['multiface_output'][0,0:42] * np.array([w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h,h])
    print('facepoints', out['multiface_output'][0,0:42])
    print('faceangle', out['multiface_output'][0,42:45])
    yaw, pitch, roll = out['multiface_output'][0,42:45]*360
    return (facepoints.astype(np.int32), yaw, pitch, roll)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    box, yaw, pitch, roll= postprocess(origimg, out)
    title =''
    for i in range(21):
       p1 = (box[i], box[i+21])
       cv2.circle(origimg, p1, 5,(0,0,213),-1)
    title = "%f, %f, %f" % (yaw, pitch, roll)
    cv2.putText(origimg, title, (10,10), cv2.FONT_ITALIC, 0.3, (0, 255, 0), 1)
    print(title)
    cv2.imshow("facepose", origimg)
 
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break

