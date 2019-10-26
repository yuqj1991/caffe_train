import shutil
import sys
import os
import random

ROOT_DIR = '../../../../../dataset/reId_data/'
reID_DataSet = ['cuhk03_release', 'Market-1501-v15.09.15']

subMarket_1501_Dir = ['gt_bbox', 'bounding_box_test']
subCUHK_03_Dir = ['labeled/train', 'labeled/val']

label = ['train_market.txt', 'train_cuhk_03.txt']


def getMark_1501_label(subDir, setfile):
	setfile_ = open(setfile, 'w')
	for img_file in os.listdir(subDir):
		label = img_file.split('_')[0]
		print(label)
		label_int = int(label)
		imgFilePath = subDir + '/' + img_file.split('.jpg')[0]
		absfullPath = os.path.abspath(imgFilePath)
		content = absfullPath + ' ' + str(label_int) + '\n'
		setfile_.write(content)
	setfile_.close()


def getCUHK_03_label(subDir, setfile):
	setfile_ = open(setfile, 'w')
	for img_file in os.listdir(subDir):
		label = img_file.split('_')[0]
		label_int = int(label)
		imgFilePath = subDir + '/' + img_file.split('.jpg')[0]
		absfullPath = os.path.abspath(imgFilePath)
		content = absfullPath +' ' + str(label_int) + '\n'
		setfile_.write(content)
	setfile_.close()
	
	

def main():
	getMark_1501_label(ROOT_DIR + 'Market-1501-v15.09.15/' + subMarket_1501_Dir[0], label[0])

if __name__ == "__main__":
	main()
