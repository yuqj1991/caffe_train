import cv2
import common
import os

SRC = '/media/scs4450/hard/luyi/tensorflow/DCGAN-tensorflow/data/cars/'
DST = '/media/scs4450/hard/luyi/tensorflow/DCGAN-tensorflow/data/cars_gray/'

def extract_image():
    lines = []
    output = os.popen('ls '+SRC+'*.jpg').readlines()
    for x in output:
        file_name = x.strip()
        lines.append(file_name)
    return lines

def convert_to_gray(files):
    for file in files:
        file_name = file.split('/')[-1]
        img = cv2.imread(SRC+file_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(DST+file_name, img_gray)
        LOG.info('Process %s Done!'%(file_name))

if __name__ == '__main__':
    LOG = common.init_my_logger()
    files = extract_image()
    convert_to_gray(files)
