import numpy as np
import shutil
import sys
import os
import xml
import cv2
import math
import random
import matplotlib.pyplot as plot
from xml.dom.minidom import Document
# from lxml.etree import Element, SubElement, tostring
sys.setrecursionlimit(1000000)

ANNOTATIONS_DIR = '../../dataset/facedata/mtfl/Annotations'
LABEL_DIR = '../../dataset/facedata/mtfl/label'
Annotation_img_dir = '../../dataset/facedata/mtfl/Annotation_img'
root_dir = "../../dataset/facedata/"
anno_mtfl_dir = ['training.txt', 'testing.txt']
thread_hold = 40 


class ConfigureHistogram(object):
	def __init__(self):
		self.count = 0
		self.width = []
		self.height = []
		self.face_count = 0
		self.face_width = []
		self.face_height = []
		self.face_blur = []
		self.face_pose = []
		self.face_occlusion = []
		self.face_illumination = []
		self.face_width_aspect_ratio = []

	def face_append(self, width, height, blur, pose, occlusion, illumination):
		self.face_count += 1
		self.face_width.append(width)
		self.face_height.append(height)
		self.face_blur.append(blur)
		self.face_pose.append(pose)
		self.face_occlusion.append(occlusion)
		self.face_illumination.append(illumination)

	def face_specfic_append(self, width, height, blur, occlusion, aspect_ratio):
		self.face_count += 1
		self.face_width.append(width)
		self.face_height.append(height)
		self.face_blur.append(blur)
		self.face_occlusion.append(occlusion)
		self.face_width_aspect_ratio.append(aspect_ratio)

	def image_append(self, img_width, img_height):
		self.count += 1
		self.width.append(img_width)
		self.height.append(img_height)


# generate label txt file for each image
def convert_src_anno_label(split_file):
	mainSetFile = open(root_dir+ "mtfl/ImageSets/Main/" +str(split_file.split("/")[-1]), "w")
	with open(split_file, 'r') as label_file:
		while True:
			img_file_info = label_file.readline().split(' ')
			if len(img_file_info) <= 2:
			    break
			img_filename = img_file_info[1]
			img_filename = img_filename.replace('\\', '/')
			img_file = root_dir + "mtfl/JPEGImages/" + img_filename
			source_img = cv2.imread(img_file)
			fullImg = os.path.abspath(img_file) + '\n'
			mainSetFile.writelines(fullImg)
			labelFile = open(LABEL_DIR+'/'+fullImg.split("/")[-1].split(".")[0], "w")
			if 1:
			    print("##################################")
			    print("imgfile path: ", img_file)
			    print("imgfile_name: ", img_filename)
			    print ("anno img file: ", Annotation_img_dir+"/"+str(img_file.split("/")[-1]))
			    print("label file path: ", labelFile)
			x1 = img_file_info[2]
			x2 = img_file_info[3]
			x3 = img_file_info[4]
			x4 = img_file_info[5]
			x5 = img_file_info[6]
			y1 = img_file_info[7]
			y2 = img_file_info[8]
			y3 = img_file_info[9]
			y4 = img_file_info[10]
			y5 = img_file_info[11]
			x =[x1, x2, x3, x4, x5];
			y =[y1, y2, y3, y4, y5];
			for ii in range(5):
				cv2.circle(source_img, (int(float(x[ii])), int(float(y[ii]))), 2, (0, 0, 225), 1)
			
			cv2.imwrite(Annotation_img_dir+"/"+str(img_file.split("/")[-1]), source_img)
			
			gender = img_file_info[12]
			glass = img_file_info[14]
			headpose = img_file_info[15]
			landmark = x1 + " " + x2 + " " + x3 + " " + x4 \
						+ " " + x5 + " " + y1 + " " + y2 + " " \
						+ " " + y3 + " " + y4 + " " + y5 + " " + gender + " " + glass + " "+ headpose
			labelFile.write(landmark)
			labelFile.close()
	mainSetFile.close()


# shuffle samples
def shuffle_file(filename):
	f = open(filename, 'r+')
	lines = f.readlines()
	random.shuffle(lines)
	f.seek(0)
	f.truncate()
	f.writelines(lines)
	f.close()


def main():

	for sub in anno_mtfl_dir:
		dir = "../../dataset/facedata/mtfl/"+sub
		convert_src_anno_label(dir)
		shuffle_file(dir)

if __name__ == '__main__':
	main()



