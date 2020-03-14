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
from xml.dom.minidom import Document
# from lxml.etree import Element, SubElement, tostring
sys.setrecursionlimit(1000000)

annotationDir = '../../../dataset/facedata/wider_face/Annotations'
labelDir = '../../../dataset/facedata/wider_face/label'
widerSetFile = ['wider_train', 'wider_val']
annoImgDir = '../../../dataset/facedata/wider_face/annoImg'
root_dir = "../../../dataset/facedata/wider_face"
widerfaceSplitDict = {'wider_train':'wider_face_train_bbx_gt.txt' , 'wider_val':'wider_face_val_bbx_gt.txt'}
height_level = [120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440,9000]
minDetectSize = 20 
classflyFile = "./wider_face_classfly_distance_data.txt"


samples = 0
samples_width = 0
samples_height = 0
min_size = 50 
max_size = 60


class ConfigureHistogram(object):
	def __init__(self):
		self.count = 0
		self.width = []
		self.height = []
		self.face_count = 0
		self.face_width = []
		self.face_height = []
		self.face_blur = []
		self.face_pose = []
		self.face_occlusion = []
		self.face_illumination = []
		self.face_width_aspect_ratio = []
		self.relative_face_image_aspect_width_ratio = []
		self.relative_face_image_aspect_height_ratio = []

	def face_append(self, width, height, blur, pose, occlusion, illumination):
		self.face_count += 1
		self.face_width.append(width)
		self.face_height.append(height)
		self.face_blur.append(blur)
		self.face_pose.append(pose)
		self.face_occlusion.append(occlusion)
		self.face_illumination.append(illumination)

	def face_specfic_append(self, width, height, aspect_ratio, width_ratio, height_ratio):
		self.face_count += 1
		self.face_width.append(width)
		self.face_height.append(height)
		self.face_width_aspect_ratio.append(aspect_ratio)
		self.relative_face_image_aspect_width_ratio.append(width_ratio)
		self.relative_face_image_aspect_height_ratio.append(height_ratio)

	def image_append(self, img_width, img_height):
		self.count += 1
		self.width.append(img_width)
		self.height.append(img_height)


def draw_hist(myList, Title, xlabel, ylabel, xmin, xmax, ymin, ymax, bins):
	plot.hist(myList, bins)
	plot.xlabel(xlabel)
	plot.xlim(xmin, xmax)
	plot.ylabel(ylabel)
	plot.ylim(ymin, ymax)
	plot.title(Title)
	plot.show()


# statsic specfic image height range ''', heightMin, heightMax'''
def collectFaceData(source_folder,label_folder, classflydataFile):
	# source_folder: ../wider_face/JPEGImages/wider_train/images
	# label_folder: ../wider_face/label
	# classflydataFile: ./wider_face_classfy_distance_data.txt
	# assert heightMax > heightMin
	# print("heightMin, %d ,heightMax %d"%(heightMin, heightMax))
	sub_image_folder = os.listdir(source_folder)
	classfly_file = open(classflydataFile, 'a+')
	static_his = ConfigureHistogram()
	for sub_folder in sub_image_folder:
		file_list = os.listdir(source_folder+'/'+sub_folder)
		for image_file in file_list:
			image_file_full_path = source_folder+'/'+sub_folder+'/'+image_file
			srcImage = cv2.imread(image_file_full_path)
			img_width = srcImage.shape[1]
			img_height = srcImage.shape[0]
			static_his.image_append(img_width, img_height)
			label_file = label_folder+'/'+image_file.split('.')[0]
			print("label_file: ", label_file)
			if not os.path.exists(label_file):
				continue
			with open(label_file, 'r') as lab_file:
				lab_an_line = lab_file.readline()
				while lab_an_line:
					anno_str_bbox = lab_an_line.split(' ')
					x_min = anno_str_bbox[0]
					y_min = anno_str_bbox[1]
					face_width = anno_str_bbox[2]
					face_height = anno_str_bbox[3]
					if int(face_width) < minDetectSize or int(face_height) < minDetectSize:
						lab_an_line = lab_file.readline()
						continue
					width_realtive_ratio = float(int(face_width)/int(img_width))
					height_realtive_ratio = float(int(face_width)/int(img_height))
					################### cluster BBox ############################
					## get relative x, y , w, h corresponind width, height#######
					################### cluster BBox ############################
					class_bdx_center_x = float((int(x_min)+int(x_min)+int(face_width))/(2*int(img_width)))
					class_bdx_center_y = float((int(y_min)+int(y_min)+int(face_height))/(2*int(img_height)))
					class_bdx_w = float(int(face_width)/int(img_width))
					class_bdx_h = float(int(face_height)/int(img_height))
					classfly_content = str(class_bdx_center_x) + ' ' + str(class_bdx_center_y) + ' ' + str(class_bdx_w)+ ' '+str(class_bdx_h)+'\n'
					classfly_file.writelines(classfly_content)
					###################### end cluster BBox ####################
					aspec_ratio =float(class_bdx_w/class_bdx_h)
					static_his.face_specfic_append(int(face_width), int(face_height),
														aspec_ratio,
														width_realtive_ratio, height_realtive_ratio)
					
					lab_an_line = lab_file.readline()
	classfly_file.close()
	return static_his


def draw_histogram_specfic_range_base_data():
	face_num=0
	####### clean classfy file ######
	classfy_ = open(classflyFile, "w")
	classfy_.truncate()
	classfy_.close()

	static_data = collectFaceData('../../../dataset/facedata/wider_face/JPEGImages/wider_train/images',
														'../../../dataset/facedata/wider_face/label',
													classflyFile)
	img_num = static_data.count
	face_num += static_data.face_count
	print('img_num: ', img_num)
	img_face_num = static_data.face_count
	# height_min and _max
	img_height_min = np.min(static_data.height)
	img_height_max = np.max(static_data.height)
	# face_width_min and _max
	face_width_min = np.min(static_data.face_width)
	face_width_max = np.max(static_data.face_width)
	# face_height_min and _max
	face_height_min = np.min(static_data.face_height)
	face_height_max = np.max(static_data.face_height)
	# face width/height aspec_ratio
	face_aspec_ratio_min = np.min(static_data.face_width_aspect_ratio)
	face_aspec_ratio_max = np.max(static_data.face_width_aspect_ratio)
	# face width image_relative_ratio
	face_width_aspec_ratio_min = np.min(static_data.relative_face_image_aspect_width_ratio)
	face_width_aspec_ratio_max = np.max(static_data.relative_face_image_aspect_width_ratio)
	# face height image_relative_ratio
	face_height_aspec_ratio_min = np.min(static_data.relative_face_image_aspect_height_ratio)
	face_height_aspec_ratio_max = np.max(static_data.relative_face_image_aspect_height_ratio)
	# print img_height and img_width(min and max), and face_height and width (min and max)
	print('img height min & max ', img_height_min, img_height_max)
	print('img_face width min & max ', face_width_min, face_width_max)
	print('img_face height min & max ', face_height_min, face_height_max)
	print('face_aspec_ratio_ min & max ', face_aspec_ratio_min, face_aspec_ratio_max)
	print('face width relative ratio min & max', face_width_aspec_ratio_min, face_width_aspec_ratio_max)
	print('face height relative ratio min & max ', face_height_aspec_ratio_min, face_height_aspec_ratio_max)
	# i want to know the img height disturibution , so i use histogram, and i need to make some gap between the min and max of img_height
	# and by the max gap between the max and the min ,i want to make 120 gap,year
	# =====================================================================
	# face width/height aspect_ratio historgram
	# =====================================================================
	draw_hist(static_data.face_height, 'face_height', 'height', 'num', 20, 960, 2000, 30000, 7)
	draw_hist(static_data.face_width, 'face_width', 'width', 'num', 20, 960, 2000, 30000, 7)
	# =====================================================================
	# face width/height aspect_ratio historgram
	# =====================================================================
	draw_hist(static_data.face_width_aspect_ratio, 'aspect_ratio', 'width/height', 'num', 0, 4, 2000, 30000, 7)
	#######################################################################
	if 0: 
		# ==========================================================
		# face_width min & max, and i will set gap between min & max
		# ==========================================================
		face_width_iter = math.ceil(float(976)/90)
		createVar = locals()
		for ii in range(int(face_width_iter)):
			createVar['face_width_levle_gap_'+str(ii*90)] =0
		for ii in range(len(static_data.face_width)):
			for jj in range(int(face_width_iter)):
				if 90*(jj) <= static_data.face_width[ii] < 90*(jj+1):
					createVar['face_width_levle_gap_'+str(jj*90)] += 1
					break
		face_width_list = []
		for ii in range(int(face_width_iter)):
			if createVar['face_width_levle_gap_'+str(ii*90)]>0:
				face_width_list.append(createVar['face_width_levle_gap_'+str(ii*90)])
				print('face_width_levle_gap_'+str(ii*90),createVar['face_width_levle_gap_'+str(ii*90)])
		# ===============================================================
		# face_height min & max, and i will set gap between min & max too
		# ===============================================================
		face_height_iter = math.ceil(float(1289)/120)
		for ii in range(int(face_height_iter)):
			createVar['face_height_levle_gap_'+str(ii*120)] =0
		for ii in range(len(static_data.face_height)):
			for jj in range(int(face_height_iter)):
				if 120*(jj) <= static_data.face_height[ii] < 120*(jj+1):
					createVar['face_height_levle_gap_'+str(jj*120)] += 1
					break
		face_height_list = []
		for ii in range(int(face_height_iter)):
			if createVar['face_height_levle_gap_'+str(ii*120)]>0:
				face_height_list.append(createVar['face_height_levle_gap_'+str(ii*120)])
				print('face_height_levle_gap_'+str(ii*120),createVar['face_height_levle_gap_'+str(ii*120)])
	print('####################################################')
	print("face_num:", face_num)


###############depricate function###############################
# recursive function for the file of all readlines, maybe killed
def generate_label_file(label_line):
	if len(label_line) == 0:
		return
	image_file_name = label_line[0]
	print("image_file_name: %s"%{image_file_name})
	image_bbox = int(label_line[1])
	label_image_file = labelDir +'/'+ image_file_name.split('.')[-2].split('/')[-1]
	lab_file = open(label_image_file,'w')
	for ii in range(image_bbox):
		index_bbox = label_line[(ii + 1) + 1]
		lab_file.writelines(index_bbox)
	lab_file.close()
	return generate_label_file(label_line[2+image_bbox:])


# generate pascal _image Set , subdirectory Main train and test , and xml file ,and annotation_img_file
def generate_pascal_image_set(widerSplit_file, wider_source_directory, sub_dir, save_folder):
	generate_annoXml_and_text_from_widerSplit(widerSplit_file, sub_dir, annotationDir)
	setFile = open(save_folder+'/'+sub_dir+'.txt', 'w')
	imageListDir = os.listdir(wider_source_directory+'/'+sub_dir+'/images')
	for imageDir in imageListDir:
		img_file_names = os.listdir(wider_source_directory+'/'+sub_dir+'/images'+'/'+imageDir)
		for img_file_ in img_file_names:
			img_no_jpg = img_file_.split('.jpg')[0]
			full_img_file_ = wider_source_directory+'/'+sub_dir+'/images'+'/'+imageDir+'/'+img_no_jpg
			if sub_dir == 'wider_train' or sub_dir == 'wider_val':
				imgfileline = os.path.abspath(full_img_file_) + '\n'
				if os.path.exists(os.path.abspath(root_dir + '/label/' + img_no_jpg)):		
					setFile.writelines(imgfileline)
	setFile.close()


# shuffle samples
def shuffle_imageSetfile(filename):
	f = open(filename, 'r+')
	lines = f.readlines()
	random.shuffle(lines)
	f.seek(0)
	f.truncate()
	f.writelines(lines)
	f.close()


# generate xml file from wider_face
def generate_annoXml_and_text_from_widerSplit(widerSplit_file, subDir, xml_save_folder):
	global samples_width
	global samples_height
	global samples
	with open(widerSplit_file, 'r') as split_file:
		while True:
			img_filename = split_file.readline()[:-1]
			if img_filename == "":
				break
			print("img_filename:%s"%{img_filename})
			imgpath = root_dir + '/JPEGImages/' + subDir + '/images/' + img_filename
			#label text file path
			label_image_file = labelDir + '/' + img_filename.split('.')[-2].split('/')[-1]
			label_file = open(label_image_file, 'w')
			#xml file path
			xml_file_path = xml_save_folder+'/'+img_filename.split('.')[-2].split('/')[-1]+'.xml'
			print('xml_file_path: %s' %{xml_file_path})
			source_img = cv2.imread(imgpath)
			doc = Document()
			annotation = doc.createElement('annotation') 	# annotation element
			doc.appendChild(annotation)
			folder = doc.createElement('folder')
			folder_name = doc.createTextNode('wider_face')
			folder.appendChild(folder_name)
			annotation.appendChild(folder)
			filename_node = doc.createElement('filename')
			filename_name = doc.createTextNode(img_filename)
			filename_node.appendChild(filename_name)
			annotation.appendChild(filename_node)
			source = doc.createElement('source')  # source sub_element
			annotation.appendChild(source)
			database = doc.createElement('database')
			database.appendChild(doc.createTextNode('wider_face Database'))
			annotation.appendChild(database)
			annotation_s = doc.createElement('annotation')
			annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
			source.appendChild(annotation_s)
			image = doc.createElement('image')
			image.appendChild(doc.createTextNode('flickr'))
			source.appendChild(image)
			flickrid = doc.createElement('flickrid')
			flickrid.appendChild(doc.createTextNode('-1'))
			source.appendChild(flickrid)
			owner = doc.createElement('owner')  # company element
			annotation.appendChild(owner)
			flickrid_o = doc.createElement('flickrid')
			flickrid_o.appendChild(doc.createTextNode('deep'))
			owner.appendChild(flickrid_o)
			name_o = doc.createElement('name')
			name_o.appendChild(doc.createTextNode('deep'))
			owner.appendChild(name_o)
			size = doc.createElement('size')   # img size info element
			annotation.appendChild(size)
			width = doc.createElement('width')
			width.appendChild(doc.createTextNode(str(source_img.shape[1])))
			height = doc.createElement('height')
			height.appendChild(doc.createTextNode(str(source_img.shape[0])))
			depth = doc.createElement('depth')
			depth.appendChild(doc.createTextNode(str(source_img.shape[2])))
			size.appendChild(width)
			size.appendChild(height)
			size.appendChild(depth)
			# split_file numbox
			numbbox = int(split_file.readline())
			for _ in range(numbbox):
				line = split_file.readline()
				anno_bbox = line.split(' ')
				x_min = anno_bbox[0]
				y_min = anno_bbox[1]
				width = anno_bbox[2]
				height = anno_bbox[3]
				blur = anno_bbox[4]
				intBlur = int(blur)
				invalid = anno_bbox[7]
				occlusion = anno_bbox[8]
				intOcclu_ = int(occlusion)
				difficult = 0
				newline = x_min + ' '+y_min+ ' '+width+ ' '+height + ' \n'
				if int(invalid) == 1 or intBlur == 2 or intOcclu_ == 2:
					continue
				if int(width)< minDetectSize or int(height)< minDetectSize:
					continue
				samples +=1
				if int(width)>= min_size and int(width)< max_size:
					samples_width  += 1
				if int(height)>= min_size and int(height)< max_size:
					samples_height +=1
				# label text file
				label_file.writelines(newline)
				# xml annotation file
				objects = doc.createElement('object')
				annotation.appendChild(objects)
				object_name = doc.createElement('name')
				object_name.appendChild(doc.createTextNode('face'))
				objects.appendChild(object_name)
				blur_node = doc.createElement('blur')
				blur_node.appendChild(doc.createTextNode(blur))
				objects.appendChild(blur_node)
				occlusion_node = doc.createElement('occlusion')
				occlusion_node.appendChild(doc.createTextNode(occlusion))
				objects.appendChild(occlusion_node)
				difficult_node = doc.createElement('difficult')
				difficult_node.appendChild(doc.createTextNode(str(difficult)))
				objects.appendChild(difficult_node)
				boundbox = doc.createElement('bndbox')  # boundbox
				objects.appendChild(boundbox)
				xmin = doc.createElement('xmin')
				xmin.appendChild(doc.createTextNode(x_min))
				boundbox.appendChild(xmin)
				ymin = doc.createElement('ymin')
				ymin.appendChild(doc.createTextNode(y_min))
				boundbox.appendChild(ymin)
				xmax = doc.createElement('xmax')
				xmax.appendChild(doc.createTextNode(str(int(x_min) + int(width))))
				boundbox.appendChild(xmax)
				ymax =doc.createElement('ymax')
				ymax.appendChild(doc.createTextNode(str(int(y_min) + int(height))))
				boundbox.appendChild(ymax)
			label_file.close()
			xml_file = open(xml_file_path, 'w')
			xml_file.write(doc.toprettyxml(indent=''))
			xml_file.close()
			data = open(label_image_file, "r").read()
			if len(data)==0:
				os.remove(label_image_file)
	split_file.close()


def main():
	# generate imagesetfile xmlfile & labelfile
	if 1:
		for sub in widerSetFile:
			splitfile = "../../../dataset/facedata/wider_face_split/"+widerfaceSplitDict[sub]
			generate_pascal_image_set(splitfile, root_dir + '/JPEGImages/', sub, root_dir+'/ImageSets/Main')
			print('samples: ', samples)
			print("samples_width: %d, and samples_height: %d"%(samples_width, samples_height))
			shuffle_imageSetfile('../../../dataset/facedata/wider_face/ImageSets/Main'+'/'+sub+'.txt')
			
	# static and get classflyFile
	if 1:
		draw_histogram_specfic_range_base_data()
if __name__ == '__main__':
	main()



