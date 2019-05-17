# -*- coding:UTF-8 -*-
from __future__ import division
import numpy as np
import shutil
import sys
import os
import xml
import cv2
import math
import random
import matplotlib.pyplot as plot

root_dir = "../../dataset/car_person_data/car_license/ccpd_dataset"
set_dir = "ImageSets/Main"
image_dir = "JPEGImages"
label_dir = "label"
lengthTestset = 50000
lengthTrain = 280001
provNum, alphaNum, adNum = 34, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def generate_label(imagefilepath, savefilepath):
	print(imagefilepath)
	img = cv2.imread(imagefilepath)
	img_h = img.shape[0]
	img_w = img.shape[1]
	labelInfo = imagefilepath.split("/")[-1].split(".jpg")[0].split("-")
	labelbndBox = labelInfo[2].split('_')
	left_upBox_x = labelbndBox[0].split("&")[0]
	left_upBox_y = labelbndBox[0].split("&")[1]
	right_bottom_x = labelbndBox[1].split("&")[0]
	right_bottom_y = labelbndBox[1].split("&")[1]
	exactbndBox = labelInfo[3].split('_')
	vertices_1_x = exactbndBox[0].split("&")[0]
	vertices_1_y = exactbndBox[0].split("&")[1]
	vertices_2_x = exactbndBox[1].split("&")[0]
	vertices_2_y = exactbndBox[1].split("&")[1]
	vertices_3_x = exactbndBox[2].split("&")[0]
	vertices_3_y = exactbndBox[2].split("&")[1]
	vertices_4_x = exactbndBox[3].split("&")[0]
	vertices_4_y = exactbndBox[3].split("&")[1]
	licenseNum = labelInfo[4].split('_')
	labelContent =" ".join(b for b in licenseNum)
	label_file_ = open(savefilepath, "w")
	label_file_.write(labelContent)
	label_file_.close()
	
	
def generate_train_setfile(imagefiledir, setfile):
	setfile_ = open(setfile, "a+")
	global lengthTrain
	for imagefilepath in os.listdir(imagefiledir):
		imgpath = imagefiledir +'/'+imagefilepath
		absimgfilepath = os.path.abspath(imgpath)
		setfile_.write(absimgfilepath+'\n')
		savefilepath = root_dir+'/'+label_dir +'/'+ absimgfilepath.split('/')[-1].split('.jpg')[0] +"_lpnumber"
		generate_label(absimgfilepath, savefilepath)
		lengthTrain += 1
	
	
def split_train_setfile(trainSetfilepath, testSetfilepath, testline, count, setfile):
	lines = []
	with open(trainSetfilepath, 'r+w') as trainfile:
		for index, line in enumerate(trainfile):
			with open(testSetfilepath, 'a+') as testfile:
				assert testline <= count
				if index== testline:
					testfile.write(line)
				else:
					lines.append(line)
			testfile.close()
		trainfile.close()
	file_ = open(setfile, 'r+w')
	file_.seek(0)
	file_.truncate()
	for linecontent in lines:
		file_.write(linecontent)
	file_.close()
	
	
def generate_random_test_indexlist(lengthTrainset, lengthTestset):
	test_length = 1
	test_list = []
	while test_length<=lengthTestset:
		random_number = random.randint(1, lengthTrainset - 1)
		if random_number not in test_list:
			test_list.append(random_number)
			test_length = len(test_list)+1
	return test_list


trainsetfilepath = root_dir + '/' + set_dir + '/training_lp.txt'
trainsetfilecopypath = root_dir + '/' + set_dir + '/training_copy_lp.txt'
testsetfilepath = root_dir + '/' + set_dir + '/testing_lp.txt'

for imgdir in os.listdir(root_dir + '/' + image_dir):
	fullimgdirpath = root_dir + '/' + image_dir +'/'+imgdir
	generate_train_setfile(fullimgdirpath, trainsetfilepath)

shutil.copyfile(trainSetfilepath, trainsetfilecopypath)

test_list = generate_random_test_indexlist(lengthTrain, lengthTestset)
numlines =0
for index in test_list:
	split_train_setfile(trainsetfilecopypath, testsetfilepath, index, lengthTrain,trainsetfilepath)
	numlines += 1

