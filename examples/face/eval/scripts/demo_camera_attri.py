import numpy as np
import argparse
import sys,os  
import cv2
caffe_root = '../../../../../caffe_train/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

maxMargin = 66
minMargin = 36

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='.prototxt file for inference', default = '../../../../net/face_detector.prototxt')
    parser.add_argument('--weights', type=str, help='.caffemodel file for inference', default = '../../../../net/face_detector.caffemodel')
    parser.add_argument('--facemodel', type=str, help='.prototxt file for inference face landmarks', default = '../../../../net/face_attributes.prototxt')
    parser.add_argument('--faceweights', type=str, help='.caffemodel file for inference face landmarks weights', default = '../../../../net/face_attributes.caffemodel')
    return parser

parser1 = make_parser()
args = parser1.parse_args()
net_file= args.model
caffe_model= args.weights
face_file= args.facemodel
face_model= args.faceweights


if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(net_file,caffe_model,caffe.TEST) 
face_net = caffe.Net(face_file,face_model,caffe.TEST) 

CLASSES = ('background', 'face')
gender_content = ('male', 'female')
#glasses_content = ('wearing glasses', 'not wearing glasses')


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
    img = img - [103.94, 116.78, 123.68] # 127.5
    img = img * 0.007843
    return img


def preprocess(src, size):
    img = cv2.resize(src, size)
    img = img - 127.5
    img = img * 0.007843
    return img


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return box.astype(np.int32), conf, cls


def postprocessface(img, out):
    h = img.shape[0]
    w = img.shape[1]
    facepoints = out['multiface_output'][0,0:10] * np.array([w, w, w, w, w, h, h, h, h, h])
    faceangle = out['multiface_output'][0,10:13]
    gender = out['multiface_output'][0,13:15]
    gender_index = np.argmax(gender)
    return facepoints.astype(np.int32), faceangle, gender_content[gender_index]


def detect():
    cap = cv2.VideoCapture(0)
    while True:
       ret, frame = cap.read()
       #frame=cv2.flip(frame,1)
       h = frame.shape[0]
       w = frame.shape[1]
       img = preprocessdet(frame, (320, 320))
       img = img.astype(np.float32)
       img = img.transpose((2, 0, 1))

       net.blobs['data'].data[...] = img
       out = net.forward()
       box, conf, cls = postprocess(frame, out)
       for i in range(len(box)):
          if conf[i]>=0.25:
             p1 = (box[i][0], box[i][1])
             p2 = (box[i][2], box[i][3])
             
             x1 = max_(0, box[i][0] - minMargin/2)
             x2 = min_(box[i][2]+minMargin/2, w)
             y1 = max_(0, box[i][1] - maxMargin/2)
             y2 = min_(box[i][3]+minMargin/2, h)
             
             p11 = (x1, y1)
             p22 = (x2, y2)
             
             ori_img = frame[y1:y2, x1:x2, :]
             ############face attributes#######################
             oimg = preprocess(ori_img, (96, 96))
             oimg = oimg.astype(np.float32)
             oimg = oimg.transpose((2, 0, 1))
             face_net.blobs['data'].data[...] = oimg
             face_out = face_net.forward()
             boxpoint, faceangle, gender = postprocessface(ori_img, face_out)
             yaw, pitch, roll = faceangle
             for jj in range(5):
                 point = (boxpoint[jj], boxpoint[jj+5])
                 cv2.circle(ori_img, point, 3, (0,0,213), -1)
             cv2.rectangle(frame, p1, p2, (0,255,0))
             p3 = (max(p1[0], 15), max(p1[1], 15))
             title = "yaw: %f, pitch: %f, roll: %f, %s" % (yaw, pitch, roll, gender)
             print(title)
             cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
       cv2.imshow("face", frame)
       k = cv2.waitKey(30) & 0xff
       if k == 27 : 
          return False

if __name__=="__main__":
    detect()
