import shutil
import sys
import os
import random
wider_directory = ['training', 'testing']
root_dir ='../../../../../face_train/examples/deepano_face/face_attributes/scripts/'
root_dataset = '../../../../../dataset/facedata/mtfl/label/'
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
				xmlline_ = root_dataset+imgname.split('.')[0]
				xmlline = os.path.abspath(xmlline_) + '\n'
				#print (os.path.abspath(xmlline_))
				#print(xmlline)
				newline = imgline.strip()+' '+xmlline
				#print(newline)
				#break
				newfile.write(newline)
		f.close()
		newfile.close()
		shuffle_file(newSetfilepath)

def main():
	generate_list("../../../../../dataset/facedata/mtfl/ImageSets/Main")

if __name__=='__main__':
	main()
