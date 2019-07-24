from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import shutil
import sys
import os
import xml
import cv2
import math
import random
import matplotlib.pyplot as plot
from scipy import misc
import argparse
from time import sleep

gender_content = ('male', 'female')
glasses_content = ('wearing glasses', 'not wearing glasses')

sampleTextSetFile = ["training.txt", "testing.txt"]
root_dir = "../../../dataset/facedata/mtfl/"
margin = 32
labelDir = 'label'
cropImgdir = 'cropImg'
if not os.path.isdir(root_dir+ cropImgdir):
	os.mkdir(root_dir+ cropImgdir)
def extractTrainLabfile(split_file):
	mainSetFile = open(root_dir+ "ImageSets/Main/" +str(split_file.split("/")[-1]), "w")
	print(mainSetFile)
	with open(split_file, 'r') as label_file:
		while True:
			img_file_info = label_file.readline().split(' ')
			if len(img_file_info) <= 2:
				break
			img_filename = img_file_info[0]
			img_filename = img_filename.replace('\\', '/')
			img_file =root_dir+ "JPEGImages/" + img_filename
			source_img = cv2.imread(img_file)
			assert source_img.shape[2]==3
			fullImg = os.path.abspath(img_file) + '\n'
			if 1:
				print("##################################")
				print("imgfile path: ", img_file)
				print("imgfile_name: ", img_filename)
				print ("anno img file: ", "annoImage/"+str(img_file.split("/")[-1]))
			xmin = int(img_file_info[1])
			xmax = int(img_file_info[2])
			ymin = int(img_file_info[3])
			ymax = int(img_file_info[4])
			x1 = float(img_file_info[5])
			y1 = float(img_file_info[6])
			x2 = float(img_file_info[7])
			y2 = float(img_file_info[8])
			x3 = float(img_file_info[9])
			y3 = float(img_file_info[10])
			x4 = float(img_file_info[11])
			y4 = float(img_file_info[12])
			x5 = float(img_file_info[13])
			y5 = float(img_file_info[14])
			if len(img_file_info) == 19:
				gender = img_file_info[15]
				glass = img_file_info[17]
			elif len(img_file_info) == 17:
				gender = img_file_info[15]
				glass = img_file_info[16]
			x =[x1, x2, x3, x4, x5]
			x_arrary = np.array(x)
			x_max = x_arrary[np.argmax(x_arrary)]
			x_min = x_arrary[np.argmin(x_arrary)]
			y =[y1, y2, y3, y4, y5]
			y_arrary = np.array(y)
			y_max = y_arrary[np.argmax(y_arrary)]
			y_min = y_arrary[np.argmin(y_arrary)]
			img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
			img_size = np.asarray(img.shape)[0:2]
			bb = np.zeros(4, dtype=np.int32)
			bb[0] = np.maximum(xmin-margin/2, 0)
			bb[1] = np.maximum(ymin-margin/2, 0)
			bb[2] = np.minimum(xmax + margin/2, img_size[1])
			bb[3] = np.minimum(ymax + margin/2, img_size[0])
			cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
			cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
			cv2.imwrite(root_dir+"annoImage/"+str(img_file.split("/")[-1]), cropped)
			x11 = x1 - bb[0]
			x22 = x2 - bb[0]
			x33 = x3 - bb[0]
			x44 = x4 - bb[0]
			x55 = x5 - bb[0]
			y11 = y1 - bb[1]
			y22 = y2 - bb[1]
			y33 = y3 - bb[1]
			y44 = y4 - bb[1]
			y55 = y5 - bb[1]
			x_a = [x11, x22, x33, x44, x55]
			x_crop = np.array(x_a).astype(np.int32)
			y_a = [y11, y22, y33, y44, y55]
			y_crop = np.array(y_a).astype(np.int32)
			if 1:
				crop_img = cv2.imread(root_dir+ "annoImage/"+str(img_file.split("/")[-1]))
				for ii in range(5):
					cv2.circle(crop_img, (x_crop[ii], y_crop[ii]), 5,(0,0,213),-1)
					title = "%s, %s" % (gender_content[int(gender) - 1], glasses_content[int(glass) - 1])
					p3 = (15, 15)
					cv2.putText(crop_img, title, p3, cv2.FONT_ITALIC, 0.3, (0, 255, 0), 1)
				cv2.imwrite(root_dir+ cropImgdir+"/"+ "crop_"+ str(img_file.split("/")[-1]), crop_img)
			labelFile = open(root_dir+ labelDir+'/'+fullImg.split("/")[-1].split(".")[0], "w")
			landmark = str(x11) + " " + str(x22) + " " + str(x33) + " " + str(x44) \
						+ " " + str(x55) + " " + str(y11) + " " + str(y22) + " " + str(y33) + " " + str(y44) + " " + str(y55) + " " + 										gender + " " + glass
			labelFile.write(landmark)
			labelFile.close()
			mainSetFile.writelines(os.path.abspath(root_dir+ "annoImage/"+str(img_file.split("/")[-1]))+'\n')
	mainSetFile.close()
	
	
for set in sampleTextSetFile:
	extractTrainLabfile(root_dir+ set)
