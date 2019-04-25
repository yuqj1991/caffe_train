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
    parser.add_argument('--facemodel', type=str, required=True, help='.prototxt file for inference face landmarks')
    parser.add_argument('--faceweights', type=str, required=True, help='.caffemodel file for inference face landmarks weights')
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
net2 = caffe.Net(face_file,face_model,caffe.TEST)  

CLASSES = ('background', 'face')
blur_classes = ('clear', 'normal', 'heavy')
occlu_classes = ('clear', 'partial', 'heavy')
gender_content = ('male', 'female')
glasses_content = ('wearing glasses', 'not wearing glasses')
headpose_content = ('left profile', 'left', 'frontal', 'right', 'right profile')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img
    

def preprocessface(src):
    img = cv2.resize(src, (96,96))
    img = img - 127.5
    img = img * 0.007843
    return img


def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    blur_max_index = out['detection_out'][0,0,:,7]
    blur_max_index = out['detection_out'][0,0,:,8]
    return (box.astype(np.int32), conf, cls, blur_max_index.astype(np.int32), blur_max_index.astype(np.int32))
    

def postprocessface(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    facepoints = out['multiface_output'][0,0:10] * np.array([w,w,w,w,w,h,h,h,h,h])
    gender = out['multiface_output'][0,10:12]
    gender_index = np.argmax(gender)
    glasses = out['multiface_output'][0,12:14]
    glasses_index = np.argmax(glasses)
    headpose = out['multiface_output'][0,14:19]
    headpose_index = np.argmax(headpose)
    return (facepoints.astype(np.int32), gender_content[gender_index], glasses_content[glasses_index], headpose_content[headpose_index])


def detect():
    cap = cv2.VideoCapture(0)
    while True:
       ret, frame = cap.read()
       img = preprocess(frame)
       img = img.astype(np.float32)
       img = img.transpose((2, 0, 1))

       net.blobs['data'].data[...] = img
       out = net.forward()
       box, conf, cls , blur, occlu= postprocess(frame, out)
       for i in range(len(box)):
          if conf[i]>=0.25:
             p1 = (box[i][0], box[i][1])
             p2 = (box[i][2], box[i][3])
             cv2.rectangle(frame, p1, p2, (0,255,0))
             
             ori_img = frame[(box[i][1]):(box[i][3]),(box[i][0]):(box[i][2]),:]
             oimg = preprocessface(ori_img)
             oimg = oimg.astype(np.float32)
             oimg = oimg.transpose((2, 0, 1))
             net2.blobs['data'].data[...] = oimg
             out2 = net2.forward()
             boxpoint, gender, glasses, headpose = postprocessface(ori_img, out2)
             for jj in range(5):
                 point = (boxpoint[jj], boxpoint[jj+5])
                 cv2.circle(ori_img, point, 3,(0,0,213),-1)
             p3 = (max(p1[0], 15), max(p1[1], 15))
             title = "%s:%.2f,  %s, %s, %s, %s, %s" % (CLASSES[int(cls[i])], conf[i],blur_classes[int(blur[i])], occlu_classes[int(occlu[i])], gender, glasses, headpose)
             cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
       cv2.imshow("facedetector", frame)
       k = cv2.waitKey(30) & 0xff
       if k == 27 : 
          return False

if __name__=="__main__":
    detect()
