import shutil
import sys
import os
import random
ccpd_directory = ['training_lp', 'testing_lp']
root_dir ='../../../../caffe_deeplearning_train/examples/deepano_liceneseplate/scripts/'
root_dataset = '../../../../dataset/car_person_data/car_license/ccpd_dataset/label/'
def shuffle_file(filename):
	f = open(filename, 'r+')
	lines = f.readlines()
	random.shuffle(lines)
	f.seek(0)
	f.truncate()
	f.writelines(lines)
	f.close()

def generate_list(imageSetDir):
	for Set in ccpd_directory:
		Setfilepath = imageSetDir+'/'+Set+'.txt'
		newSetfilepath = root_dir + Set+'.txt'
		newfile = open(newSetfilepath, 'w')
		with open(Setfilepath,'r') as f:
			while True:
				imgline = f.readline()
				if imgline =='':
					break
				imgname =imgline.split('/')[-1].strip()
				xmlline_ = root_dataset + imgname.split('.')[0]
				xmlline = os.path.abspath(xmlline_) + '\n'
				newline = imgline.strip()+' '+xmlline
				newfile.write(newline)
		f.close()
		newfile.close()
		shuffle_file(newSetfilepath)
def main():
	generate_list("../../../../dataset/car_person_data/car_license/ccpd_dataset/ImageSets/Main")

if __name__=='__main__':
	main()
