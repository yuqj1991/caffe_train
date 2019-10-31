import shutil
import sys
import os
import random
wider_directory = ['wider_train', 'wider_val']
root_dir ='../../../../../caffe_train/examples/face/detector/scripts/'
#root_dataset = '../../../../../dataset/facedata/wider_face/Annotations/'
root_dataset = '../../../../../dataset/facedata/wider_face_add_lm_10_10/Annotations/'
ROOT_IMAGE_DIR = "../../../../../dataset/facedata/wider_face_add_lm_10_10/JPEGImages/"
def shuffle_file(filename):
	f = open(filename, 'r+')
	lines = f.readlines()
	random.shuffle(lines)
	f.seek(0)
	f.truncate()
	f.writelines(lines)
	f.close()

def generate_list(imageSetDir):
	for Set in wider_directory:
		Setfilepath = imageSetDir+'/'+Set+'.txt'
		newSetfilepath = root_dir + Set+'.txt'
		newfile = open(newSetfilepath, 'w')
		with open(Setfilepath,'r') as f:
			while True:
				imgline = f.readline()
				if imgline =='':
					break
				imgname =imgline.split('/')[-1].strip()
				xmlline_ = root_dataset+imgname+'.xml'
				xmlline = os.path.abspath(xmlline_) + '\n'
				if os.path.exists(xmlline_):
					newline = os.path.abspath(ROOT_IMAGE_DIR+imgline.strip().replace('stive', 'deep')+'.jpg')+' '+xmlline
					newfile.write(newline)
		f.close()
		newfile.close()
		shuffle_file(newSetfilepath)

def main():
	#generate_list("../../../../../dataset/facedata/wider_face/ImageSets/Main")
	generate_list("../../../../../dataset/facedata/wider_face_add_lm_10_10/ImageSets/Main")

if __name__=='__main__':
	main()
