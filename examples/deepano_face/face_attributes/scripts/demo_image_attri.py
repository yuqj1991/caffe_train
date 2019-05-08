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
gender_content = ('male', 'female')
glasses_content = ('wearing glasses', 'not wearing glasses')
headpose_content = ('left profile', 'left', 'frontal', 'right', 'right profile')
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
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    box, gender, glasses, headpose = postprocess(origimg, out)
    title =''
    for i in range(5):
       p1 = (box[i], box[i+5])
       cv2.circle(origimg, p1, 5,(0,0,213),-1)
    title = "%s, %s, %s" % (gender_content[gender], glasses_content[glasses], headpose_content[headpose])
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

