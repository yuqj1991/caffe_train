import os, sys
import shutil
import random

ROOT_DIR = '../../../../../dataset/reId_data/'

train_val_set = ['train', 'val']
subMarket_1501_Dir = ['gt_bbox', 'bounding_box_test']
subCUHK_03_Dir = ['labeled/train', 'labeled/val']
#subDukeMTMC-reID_Dir = ['', '']


reID_DataSet = ['Market-1501-v15.09.15', 'cuhk03_release']
subDataSetList = [subMarket_1501_Dir, subCUHK_03_Dir]
DirPrefixlist = ['market_1501', 'cuhk03']
combineDataDir = 'combineData'


def creatSubDataBaseDir(subDir, combineDir, subcombineDir, DirPrefix):
	if not os.path.exists(combineDir):
		os.mkdir(combineDir)
	if not os.path.exists(combineDir + '/' +subcombineDir):
		os.mkdir(combineDir + '/' +subcombineDir)
	for img_file in os.listdir(subDir):
		print(img_file)
		label = img_file.split('_')[0]
		sublabelDir = combineDir + '/' +subcombineDir + '/' + DirPrefix + '_' + label
		if not os.path.exists(sublabelDir):
			os.mkdir(sublabelDir)
		absimgfile = subDir + '/' + img_file
		shutil.copy(absimgfile, sublabelDir)


def main():
	for ii in range(len(train_val_set)):
		for idx in range(len(reID_DataSet)):
			subDir = ROOT_DIR + reID_DataSet[idx] + '/' + subDataSetList[idx][ii]
			combineDir = ROOT_DIR + combineDataDir
			subcombineDir = train_val_set[ii]
			DirPrefix = DirPrefixlist[idx]
			creatSubDataBaseDir(subDir, combineDir, subcombineDir, DirPrefix)


if __name__ == '__main__':
	main()
