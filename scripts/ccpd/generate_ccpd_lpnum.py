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
ccpd_anno_img_dir = "../../../dataset/car_person_data/car_license/ccpd_dataset/annoImg"
if not os.path.exists(ccpd_anno_img_dir):
	os.makedirs(ccpd_anno_img_dir)
set_dir = "ImageSets/Main"
image_dir = "JPEGImages"
label_dir = "label"
lengthTest = 50000
lengthTrain = 0
provNum, alphaNum, adNum = 34, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60, "W": 61, "X": 62, "Y": 63, "Z": 64}


def get_label(name):
	lpr_str = name.split('_')
	label = []
	for s in lpr_str:
		label.append(index[s])
	return np.array(label)


def convertIndex2label(licenseNum):
	fist = provinces[licenseNum[0]]
	second = alphabets[licenseNum[1]]
	third = ads[licenseNum[2]]
	forth = ads[licenseNum[3]]
	fifth = ads[licenseNum[4]]
	sixth = ads[licenseNum[5]]
	seventh = ads[licenseNum[6]]
	return fist + "_" + second + "_" + third + "_" + forth + "_" + fifth + "_" + sixth + "_" + seventh


def generate_label(imagefilepath, labelfilepath):
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
	cropped = img[(int(left_upBox_y) - 22):(int(right_bottom_y)+22), (int(left_upBox_x) -22):(int(right_bottom_x) + 22), :]
	cv2.imwrite(ccpd_anno_img_dir+"/"+"crop_"+str(imagefilepath.split("/")[-1]), cropped)
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
	with open(labelfilepath, "w") as label_file_:
		label_file_.write(labelContent)
		label_file_.close()


def generate_anothor_label(imagefilepath, labelfilepath, dirprefix):
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
	cropped = img[(int(left_upBox_y) - 22):(int(right_bottom_y)+22), (int(left_upBox_x) -22):(int(right_bottom_x) + 22), :]
	cv2.imwrite(ccpd_anno_img_dir+'/crop_' + dirprefix + str(imagefilepath.split("/")[-1]), cropped)
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
	license = []
	for idnum in licenseNum:
		license.append(int(idnum))
	numberString = convertIndex2label(license)
	numbers = get_label(numberString)
	labelContent =" ".join(str(b) for b in numbers)
	with open(labelfilepath, "w") as label_file_:
		label_file_.write(labelContent)
		label_file_.close()


def generate_setfile(imagefiledir, setfile, dirprefix):
	setfile_ = open(setfile, "a+")
	global lengthTrain
	for imagefilepath in os.listdir(imagefiledir):
		imgpath = imagefiledir +'/'+imagefilepath
		absimgfilepath = os.path.abspath(imgpath)
		annopath = os.path.abspath(ccpd_anno_img_dir)+'/crop_' + dirprefix + str(absimgfilepath.split("/")[-1])
		setfile_.write(annopath + '\n')
		labelfilepath = root_dir + '/' + label_dir + '/crop_' + dirprefix + absimgfilepath.split('/')[-1].split('.jpg')[0]
		#generate_label(absimgfilepath, labelfilepath)
		generate_anothor_label(absimgfilepath, labelfilepath, dirprefix)
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
	trainsetfilepath = root_dir + '/' + set_dir + '/training_lp.txt'
	testsetfilepath= root_dir + '/' + set_dir + '/testing_lp.txt'
	
	if 1:
		for imgdir in os.listdir(root_dir + '/' + image_dir):
			fullimgdir = root_dir + '/' + image_dir +'/'+imgdir
			generate_setfile(fullimgdir, trainsetfilepath, imgdir)

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

if __name__ == '__main__':
	main()

