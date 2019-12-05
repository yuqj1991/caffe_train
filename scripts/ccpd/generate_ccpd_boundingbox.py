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

root_dir = "../../../dataset/car_person_data/car_license/ccpd_dataset"
set_dir = "ImageSets/Main"
image_dir = "JPEGImages"
label_dir = "label"
lengthTest = 50000
lengthTrain = 280001
provNum, alphaNum, adNum = 34, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

classflyFile = "./ccpd_classfly_distance_data.txt"
collectBoxData = True


def generate_label(imagefilepath, savefilepath, classflydataFile):
	print(imagefilepath)
	classfly_file = open(classflydataFile, 'a+')
	img = cv2.imread(imagefilepath)
	img_h = img.shape[0]
	img_w = img.shape[1]
	labelInfo = imagefilepath.split("/")[-1].split(".jpg")[0].split("-")
	labelbndBox = labelInfo[2].split('_')
	left_upBox_x = labelbndBox[0].split("&")[0]
	left_upBox_y = labelbndBox[0].split("&")[1]
	right_bottom_x = labelbndBox[1].split("&")[0]
	right_bottom_y = labelbndBox[1].split("&")[1]
	if collectBoxData:
		class_bdx_center_x = float((int(left_upBox_x)+int(right_bottom_x))/(2*int(img_w)))
		class_bdx_center_y = float((int(left_upBox_y)+int(right_bottom_y))/(2*int(img_h)))
		class_bdx_w = float((int(right_bottom_x) - int(left_upBox_x))/int(img_w))
		class_bdx_h = float((int(right_bottom_y) - int(left_upBox_y))/int(img_h))
		classfly_content = str(class_bdx_center_x) + ' ' + str(class_bdx_center_y) + ' ' + str(class_bdx_w)+ ' '+str(class_bdx_h)+'\n'
		classfly_file.writelines(classfly_content)
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
	labelContent = " ".join(a for ii in range(2) for a in labelbndBox[ii].split("&"))
	label_file_ = open(savefilepath, "w")
	label_file_.write(labelContent)
	label_file_.close()
	classfly_file.close()


def generate_setfile(imagefiledir, setfile):
	setfile_ = open(setfile, "a+")
	classfy_ = open(classflyFile, "w")
	classfy_.truncate()
	classfy_.close()
	global lengthTrain
	for imagefilepath in os.listdir(imagefiledir):
		imgpath = imagefiledir +'/'+imagefilepath
		absimgfilepath = os.path.abspath(imgpath)
		setfile_.write(absimgfilepath+'\n')
		savefilepath = root_dir+'/'+label_dir +'/'+ absimgfilepath.split('/')[-1].split('.jpg')[0]
		generate_label(absimgfilepath, savefilepath, classflyFile)
		lengthTrain += 1


def split_setfile(trainSetfilepath, valList):
	valSetContent = []
	trainSetContent = []
	with open(trainSetfilepath, 'r') as setfile_:
		setContent = setfile_.readlines()
		print("setContent  length: ", len(setContent))
		for index in valList:
			valSetContent.append(setContent[index])
			setContent[index] = "a\n"
		for content in setContent:
			if content != "a\n":
				trainSetContent.append(content)
	return trainSetContent, valSetContent


def generate_random_val_list(lengthTrainSet, lengthTestSet):
	val_length = 0
	val_list = []
	while val_length <= lengthTestSet:
		random_number = random.randint(0, lengthTrainSet - 1)
		if random_number not in val_list:
			val_list.append(random_number)
			val_length = len(val_list) + 1
	return val_list


def main():
	trainsetfilepath = root_dir + '/' + set_dir + '/training.txt'
	testsetfilepath = root_dir + '/' + set_dir + '/testing.txt'

	for imgdir in os.listdir(root_dir + '/' + image_dir):
		fullimgdirpath = root_dir + '/' + image_dir +'/'+imgdir
		generate_setfile(fullimgdirpath, trainsetfilepath)

	val_list = generate_random_val_list(lengthTrain, lengthTest)
	train, val = split_setfile(trainsetfilepath, val_list)
	print("$$$$$$$$$$$write to file$$$$$$$$$$$$$$$$$$$$$$$")
	print("train set : ", len(train))
	print("val set : ", len(val))
	with open(trainsetfilepath, "w") as train_file_:
		train_file_.truncate()
		for line in train:
			train_file_.writelines(line)
		train_file_.close()
	with open(testsetfilepath, "w") as test_file_:
		for line in val:
			test_file_.writelines(line)
		test_file_.close()


if __name__ == "__main__":
	main()

