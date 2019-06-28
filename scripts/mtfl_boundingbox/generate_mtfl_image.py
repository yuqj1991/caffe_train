from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import shutil
import sys
import os
import xml
import cv2
import math
import random
import matplotlib.pyplot as plot
from scipy import misc
import argparse
import tensorflow as tf
import detect_face
from time import sleep

LABEL_DIR = '../../../dataset/facedata/mtfl/label'
annoImg_dir = '../../../dataset/facedata/mtfl/annoImage'
root_dir = "../../../dataset/facedata/"
anno_mtfl_dir = ['training.txt', 'testing.txt']


# generate label txt file for each image
def convert_src_anno_label(split_file, args, pnet, rnet, onet, minsize, threshold, factor):
	mainSetFile = open(root_dir+ "mtfl/ImageSets/Main/" +str(split_file.split("/")[-1]), "w")
	with open(split_file, 'r') as label_file:
		while True:
			img_file_info = label_file.readline().split(' ')
			if len(img_file_info) <= 2:
			    break
			img_filename = img_file_info[1]
			img_filename = img_filename.replace('\\', '/')
			img_file = root_dir + "mtfl/JPEGImages/" + img_filename
			source_img = cv2.imread(img_file)
			assert source_img.shape[2]==3
			fullImg = os.path.abspath(img_file) + '\n'
			if 1:
				print("##################################")
				print("imgfile path: ", img_file)
				print("imgfile_name: ", img_filename)
				print ("anno img file: ", annoImg_dir+"/"+str(img_file.split("/")[-1]))
				#print("label file path: ", labelFile)
			x1 = float(img_file_info[2])
			x2 = float(img_file_info[3])
			x3 = float(img_file_info[4])
			x4 = float(img_file_info[5])
			x5 = float(img_file_info[6])
			y1 = float(img_file_info[7])
			y2 = float(img_file_info[8])
			y3 = float(img_file_info[9])
			y4 = float(img_file_info[10])
			y5 = float(img_file_info[11])
			gender = img_file_info[12]
			glass = img_file_info[14]
			x =[x1, x2, x3, x4, x5]
			x_arrary = np.array(x)
			x_max = x_arrary[np.argmax(x_arrary)]
			x_min = x_arrary[np.argmin(x_arrary)]
			y =[y1, y2, y3, y4, y5]
			y_arrary = np.array(y)
			y_max = y_arrary[np.argmax(y_arrary)]
			y_min = y_arrary[np.argmin(y_arrary)]
			################################
			img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
			bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
			nrof_faces = bounding_boxes.shape[0]
			if nrof_faces>0:
				det = bounding_boxes[:,0:4]
				det_arr = []
				img_size = np.asarray(img.shape)[0:2]
				if nrof_faces>1:
					if args.detect_multiple_faces:
						for i in range(nrof_faces):
							det_arr.append(np.squeeze(det[i]))
				else:
					det_arr.append(np.squeeze(det))
				for i, det in enumerate(det_arr):
					det = np.squeeze(det)
					if det[0] <= x_min and det[1] <= y_min and det[2] >=x_max and det[3] >=y_max:
						bb = np.zeros(4, dtype=np.int32)
						bb[0] = np.maximum(det[0]-args.margin/2, 0)
						bb[1] = np.maximum(det[1]-args.margin/2, 0)
						bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
						bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
						cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
						cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
						#nrof_successfully_aligned += 1
						cv2.imwrite(annoImg_dir+"/"+str(img_file.split("/")[-1]), cropped)
						x11 = x1 - bb[0]
						x22 = x2 - bb[0]
						x33 = x3 - bb[0]
						x44 = x4 - bb[0]
						x55 = x5 - bb[0]
						y11 = y1 - bb[1]
						y22 = y2 - bb[1]
						y33 = y3 - bb[1]
						y44 = y4 - bb[1]
						y55 = y5 - bb[1]
						x_a = [x11, x22, x33, x44, x55]
						x_crop = np.array(x_a).astype(np.int32)
						y_a = [y11, y22, y33, y44, y55]
						y_crop = np.array(y_a).astype(np.int32)
						if 1:
							crop_img = cv2.imread(annoImg_dir+"/"+str(img_file.split("/")[-1]))
							for ii in range(5):
								cv2.circle(crop_img, (x_crop[ii], y_crop[ii]), 5,(0,0,213),-1)
							cv2.imwrite(annoImg_dir+"/"+ "crop_"+ str(img_file.split("/")[-1]), crop_img)
						labelFile = open(LABEL_DIR+'/'+fullImg.split("/")[-1].split(".")[0], "w")
						landmark = str(x11) + " " + str(x22) + " " + str(x33) + " " + str(x44) \
									+ " " + str(x55) + " " + str(y11) + " " + str(y22) + " " + str(y33) + " " + str(y44) + " " + str(y55) + " " + 										gender + " " + glass
						labelFile.write(landmark)
						labelFile.close()
						mainSetFile.writelines(os.path.abspath(annoImg_dir+"/"+str(img_file.split("/")[-1]))+'\n')
						################################
	mainSetFile.close()


# shuffle samples
def shuffle_file(filename):
	f = open(filename, 'r+')
	lines = f.readlines()
	random.shuffle(lines)
	f.seek(0)
	f.truncate()
	f.writelines(lines)
	f.close()


def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--margin', type=int,
		help='Margin for the crop around the bounding box (height, width) in pixels.', default=66)
	parser.add_argument('--gpu_memory_fraction', type=float,
		help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.1)
	parser.add_argument('--detect_multiple_faces', type=bool,
						help='Detect and align multiple faces per image.', default=True)
	return parser.parse_args(argv)


def main(args):
	print('Creating networks and loading parameters')
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
	minsize = 20 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709 # scale factor
	for sub in anno_mtfl_dir:
		dir = "../../../dataset/facedata/mtfl/"+sub
		convert_src_anno_label(dir, args, pnet, rnet, onet, minsize, threshold, factor)
		shuffle_file(dir)
	#print(nrof_successfully_aligned)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))



