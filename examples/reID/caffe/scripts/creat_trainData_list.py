import shutil
import sys
import os
import random

ROOT_DIR = '../../../../../dataset/reId_data/'

combineDataDir = 'combineData'
train_val_set = ['train', 'val']
labelIndex = 0


def shuffle_file(filename):
	f = open(filename, 'r+')
	lines = f.readlines()
	random.shuffle(lines)
	f.seek(0)
	f.truncate()
	f.writelines(lines)
	f.close()


def getCombineLabel(subDir,setfile):
	setfile_ = open(setfile, 'w')
	for sublabelDir in os.listdir(subDir):
		print(sublabelDir)
		for img_file in os.listdir(subDir + '/' + sublabelDir):
			print(img_file)
			global labelIndex
			imgFilePath = subDir + '/' +sublabelDir + '/' + img_file.split('.jpg')[0]
			absfullPath = os.path.abspath(imgFilePath)
			content = absfullPath + ' ' + str(labelIndex) + '\n'
			setfile_.write(content)
		labelIndex += 1
	setfile_.close()


def main():
	getCombineLabel(ROOT_DIR + combineDataDir + '/' + train_val_set[0], train_val_set[0]+'.txt')
	shuffle_file(train_val_set[0]+'.txt')

if __name__ == "__main__":
	main()
