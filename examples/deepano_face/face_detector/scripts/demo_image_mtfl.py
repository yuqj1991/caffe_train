import numpy as np
import argparse
import sys,os  
import cv2
caffe_root = '../../../../../caffe_deeplearning_train/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
mtfl_dir = "../../../../../dataset/facedata/mtfl/JPEGImages/"
mtfl_dir_set = "../../../../../dataset/facedata/mtfl/"
dataset = ["training.txt", "testing.txt"]
src_label_dir ="../../../../../dataset/facedata/mtfl/label/"
annoImg_dir = "../../../../../dataset/facedata/mtfl_crop/annoImage/"

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file for inference')
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

CLASSES = ('background',
           'face')


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
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    box, conf, cls = postprocess(origimg, out)
    newimgfile_name = imgfile.split('/')[-1]
    saveimgfilepath = "../../../../../dataset/facedata/mtfl_crop/JPEGImages/"+imgfile.split('/')[-2]+"/"+newimgfile_name
    src_label_path = src_label_dir+newimgfile_name.split(".jpg")[0]
    save_label_path = "../../../../../dataset/facedata/mtfl_crop/label/"+newimgfile_name.split(".jpg")[0]
        
    for i in range(1):
       if conf[i]>=0.25:
           print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
           p1 = (box[i][0], box[i][1])
           p2 = (box[i][2], box[i][3])
           ori_img = origimg[(box[i][1]):(box[i][3]),(box[i][0]):(box[i][2]),:]
           cv2.imwrite(saveimgfilepath, ori_img)
           with open(src_label_path, "r") as label_file_:
               while True:
                   annoInfo = label_file_.readline().split(' ')
                   if len(annoInfo)<=2:
                       break
                   x1 = annoInfo[0]
                   x2 = annoInfo[1]
                   x3 = annoInfo[2]
                   x4 = annoInfo[3]
                   x5 = annoInfo[4]
                   y1 = annoInfo[5]
                   y2 = annoInfo[6]
                   y3 = annoInfo[7]
                   y4 = annoInfo[8]
                   y5 = annoInfo[9]
                   gender = annoInfo[10]
                   glass = annoInfo[11]
                   headpose = annoInfo[12]
                   x11 = np.float32(x1)-box[i][0]
                   x22 = np.float32(x2)-box[i][0]
                   x33 = np.float32(x3)-box[i][0]
                   x44 = np.float32(x4)-box[i][0]
                   x55 = np.float32(x5)-box[i][0]
                   y11 = np.float32(y1)-box[i][1]
                   y22 = np.float32(y2)-box[i][1]
                   y33 = np.float32(y3)-box[i][1]
                   y44 = np.float32(y4)-box[i][1]
                   y55 = np.float32(y5)-box[i][1]
                   x = [x11, x22, x33, x44, x55]
                   y = [y11, y22, y33, y44, y55]
                   new_img = cv2.imread(saveimgfilepath)
                   for ii in range(5):
                       point = (int(x[ii]), int(y[ii]))
                       cv2.circle(new_img, point, 3,(0,0,213),-1)
                   cv2.imwrite(annoImg_dir+newimgfile_name, new_img)
                   save_content = str(x11) + ' '+str(x22) + ' '+str(x33) + ' '+str(x44) + ' '+str(x55) + ' '+str(y11) + ' '+str(y22) + ' '+str(y33) + ' '+str(y44) + ' '+str(y55) + ' '+gender+' '+glass+' '+headpose
                   save_file_ = open(save_label_path,"w")
                   save_file_.write(save_content)
                   save_file_.close()             
           label_file_.close()
'''           
           cv2.rectangle(origimg, p1, p2, (0,255,0))
           p3 = (max(p1[0], 15), max(p1[1], 15))
           title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
           cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("facedetector", origimg)
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
'''
    #return True
           
    
    
for sub in dataset:
    subdatafile = mtfl_dir_set+sub
    with open(subdatafile, 'r') as setfile_:
        while True:
            img_file_info = setfile_.readline().split(' ')
            if len(img_file_info) <= 2:
                break
            img_filename = mtfl_dir+img_file_info[1]
            img_filename = img_filename.replace('\\', '/')
            print(img_filename)
            detect(img_filename)
