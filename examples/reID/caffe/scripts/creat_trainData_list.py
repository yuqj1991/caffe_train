import shutil
import sys
import os
import random

ROOT_DIR = '../../../../../dataset/reId_data/'

combineDataDir = 'combineData'
train_val_set = ['train', 'val']
labelIndex = 0

vgglabmap_train = 'labelmap.txt'

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


def generat_labelmap(img_dir):
	dirs = os.listdir(img_dir)
	label = 0
	_str = ""
	for dir in dirs:
		if os.path.isdir(os.path.join(img_dir, dir)):
			content_labelmap_str = dir + ' ' + str(label) + "\n"
			print(content_labelmap_str)
			_str += content_labelmap_str
			label += 1
	return _str


def main():
	if 0:
		getCombineLabel(ROOT_DIR + combineDataDir + '/' + train_val_set[0], train_val_set[0]+'.txt')
		shuffle_file(train_val_set[0]+'.txt')
	else:
		labels_str = generat_labelmap(img_dir)
		with open(vgglabmap_train, "w") as labels_file:
			labels_file.writelines(labels_str)
		shuffle_file(vgglabmap_train)

if __name__ == "__main__":
	main()
