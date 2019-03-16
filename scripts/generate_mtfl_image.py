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

ANNOTATIONS_DIR = '../../dataset/facedata/mtfl/Annotations'
LABEL_DIR = '../../dataset/facedata/mtfl/label'
wider_directory = ['wider_train', 'wider_test', 'wider_val']
Annotation_img_dir = '../../dataset/facedata/mtfl/Annotation_img'
root_dir = "../../dataset/facedata/"
anno_mtfl_dir = ['training.txt', 'testing.txt']
height_level = [120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440,9000]
thread_hold = 40  # map:59.35%, thread_hold=40


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

	def face_append(self, width, height, blur, pose, occlusion, illumination):
		self.face_count += 1
		self.face_width.append(width)
		self.face_height.append(height)
		self.face_blur.append(blur)
		self.face_pose.append(pose)
		self.face_occlusion.append(occlusion)
		self.face_illumination.append(illumination)

	def face_specfic_append(self, width, height, blur, occlusion, aspect_ratio):
		self.face_count += 1
		self.face_width.append(width)
		self.face_height.append(height)
		self.face_blur.append(blur)
		self.face_occlusion.append(occlusion)
		self.face_width_aspect_ratio.append(aspect_ratio)

	def image_append(self, img_width, img_height):
		self.count += 1
		self.width.append(img_width)
		self.height.append(img_height)


def static_histogram_data_from_all_sample(source_folder,label_folder):
	# source_folder: ../wider_face/JPEGImages/wider_train/images
	# label_folder: ../wider_face/label
	sub_image_folder = os.listdir(source_folder)
	static_his = ConfigureHistogram()
	for sub_folder in sub_image_folder:
		file_list = os.listdir(source_folder+'/'+sub_folder)
		for image_file in file_list:
			image_file_full_path = source_folder+'/'+sub_folder+'/'+image_file
			# print(image_file_full_path)
			xml_file = xml.dom.minidom.parse('../../dataset/facedata/wider_face/Annotations/'+image_file.split('.')[-2]+'.xml')
			# print('../wider_face/Annotations/'+image_file.split('.')[-2]+'.xml')
			root = xml_file.documentElement
			width = root.getElementsByTagName('width')
			img_width = int(width[0].firstChild.data)
			# print('img_width', int(img_width.firstChild.data))
			height = root.getElementsByTagName('height')
			img_height = int(height[0].firstChild.data)
			# print('height,width', img_height,img_width)
			static_his.image_append(img_width, img_height)
			label_file = label_folder+'/'+image_file.split('.')[-2]
			with open(label_file, 'r') as lab_file:
				lab_an_line = lab_file.readline()
				while lab_an_line:
					anno_str_bbox = lab_an_line.split(' ')
					x_min = anno_str_bbox[0]
					y_min = anno_str_bbox[1]
					face_width = anno_str_bbox[2]
					face_height = anno_str_bbox[3]
					blur = anno_str_bbox[4]
					illumination = anno_str_bbox[6]
					invalid = anno_str_bbox[7]
					occlusion = anno_str_bbox[8]
					pose = anno_str_bbox[9]
					static_his.face_append(int(face_width), int(face_height), int(blur), int(pose), int(occlusion), int(illumination))
					if int(face_width)<=0 or int(face_height)<=0:
						print('face_width, and height:',int(face_width), int(face_height))
						print('../../dataset/facedata/wider_face/Annotations/'+image_file.split('.')[-2]+'.xml')
					lab_an_line = lab_file.readline()
			if img_height > 4500:
				print('../../dataset/facedata/wider_face/Annotations/'+image_file.split('.')[-2]+'.xml')
				print('img_height:', img_height)
	return static_his


# statsic specfic image height range
def statsic_specfic_face_base_data(source_folder,label_folder, heightMin, heightMax):
	# source_folder: ../wider_face/JPEGImages/wider_train/images
	# label_folder: ../wider_face/label
	assert heightMax > heightMin
	print("heightMin, %d ,heightMax %d"%(heightMin, heightMax))
	sub_image_folder = os.listdir(source_folder)
	static_his = ConfigureHistogram()
	for sub_folder in sub_image_folder:
		file_list = os.listdir(source_folder+'/'+sub_folder)
		for image_file in file_list:
			image_file_full_path = source_folder+'/'+sub_folder+'/'+image_file
			xml_file = xml.dom.minidom.parse('../../dataset/facedata/wider_face/Annotations/'+image_file.split('.')[-2]+'.xml')
			root = xml_file.documentElement
			width = root.getElementsByTagName('width')
			img_width = int(width[0].firstChild.data)
			height = root.getElementsByTagName('height')
			img_height = int(height[0].firstChild.data)
			if img_height >=heightMin and img_height < heightMax:
				static_his.image_append(img_width, img_height)
				label_file = label_folder+'/'+image_file.split('.')[-2]
				with open(label_file, 'r') as lab_file:
					lab_an_line = lab_file.readline()
					while lab_an_line:
						anno_str_bbox = lab_an_line.split(' ')
						x_min = anno_str_bbox[0]
						y_min = anno_str_bbox[1]
						face_width = anno_str_bbox[2]
						face_height = anno_str_bbox[3]
						blur = anno_str_bbox[4]
						occlusion = anno_str_bbox[5]
						aspec_ratio =float(int(face_width)/int(face_height))
						static_his.face_specfic_append(int(face_width), int(face_height), int(blur), int(occlusion), aspec_ratio)
						if int(face_width)<=0 or int(face_height)<=0:
							print('face_width, and height:',int(face_width), int(face_height))
							print('../../dataset/facedata/wider_face/Annotations/'+image_file.split('.')[-2]+'.xml')
						lab_an_line = lab_file.readline()
	return static_his


def draw_histogram_specfic_range_base_data():
	face_num=0
	for ii in range(len(height_level)):
		if ii == len(height_level)-1:
			break
		static_data = statsic_specfic_face_base_data('../../dataset/facedata/wider_face/JPEGImages/wider_train/images', '../../dataset/facedata/wider_face/label',
													 height_level[ii], height_level[ii+1])
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
		# print img_height and img_width(min and max), and face_height and width (min and max)
		print('img height min & max ', img_height_min, img_height_max)
		print('img_face width min & max ', face_width_min, face_width_max)
		print('img_face height min & max ', face_height_min, face_height_max)
		print('face_aspec_ratio_ min & max ', face_aspec_ratio_min, face_aspec_ratio_max)
		# i want to know the img height disturibution , so i use histogram, and i need to make some gap between the min and max of img_height
		# and by the max gap between the max and the min ,i want to make 120 gap,year
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


def autolabel(rects):
	for rect in rects:
		height = rect.get_height()
		plot.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))


def draw_histogram_base_data():
	static_data = static_histogram_data_from_all_sample('../../dataset/facedata/wider_face/JPEGImages/wider_train/images', '../../dataset/facedata/wider_face/label')
	img_num = static_data.count
	img_face_num = static_data.face_count
	# width_min and _max
	img_width_min = np.min(static_data.width)
	img_width_max = np.max(static_data.width)
	# height_min and _max
	img_height_min = np.min(static_data.height)
	img_height_max = np.max(static_data.height)
	# face_width_min and _max
	face_width_min = np.min(static_data.face_width)
	face_width_max = np.max(static_data.face_width)
	# face_height_min and _max
	face_height_min = np.min(static_data.face_height)
	face_height_max = np.max(static_data.face_height)
	# face blur
	img_face_blur_0 = 0
	img_face_blur_1 = 0
	img_face_blur_2 = 0
	# face pose
	img_face_pose_0 = 0
	img_face_pose_1 = 0
	# face illumination
	img_face_illumination_0 = 0
	img_face_illumination_1 = 0
	# face occlusion
	img_face_occlusion_0 = 0
	img_face_occlusion_1 = 0
	img_face_occlusion_2 = 0
	for ii in range(img_face_num):
		# blur
		if static_data.face_blur[ii] == 0:
			img_face_blur_0 += 1
		elif static_data.face_blur[ii] == 1:
			img_face_blur_1 += 1
		elif static_data.face_blur[ii] == 2:
			img_face_blur_2 += 1
		# pose
		if static_data.face_pose[ii] == 0:
			img_face_pose_0 += 1
		elif static_data.face_pose[ii] == 1:
			img_face_pose_1 += 1
		# illumination
		if static_data.face_illumination[ii] == 0:
			img_face_illumination_0 += 1
		elif static_data.face_illumination[ii] == 1:
			img_face_illumination_1 += 1
		# occlusion
		if static_data.face_occlusion[ii] == 0:
			img_face_occlusion_0 += 1
		elif static_data.face_occlusion[ii] == 1:
			img_face_occlusion_1 += 1
		elif static_data.face_occlusion[ii] == 2:
			img_face_occlusion_2 += 1
	# draw_img
	total_blur = [img_face_blur_0, img_face_blur_1, img_face_blur_2]
	total_pose = [img_face_pose_0, img_face_pose_1]
	total_illumination = [img_face_illumination_0, img_face_illumination_1]
	total_occlusion = [img_face_occlusion_0, img_face_occlusion_1, img_face_occlusion_2]
	print('img_face_blur_0, img_face_blur_1, img_face_blur_2', img_face_blur_0, img_face_blur_1, img_face_blur_2)
	print('img_face_pose_0, img_face_pose_1', img_face_pose_0, img_face_pose_1)
	print('img_face_illumination_0, img_face_illumination_1', img_face_illumination_0, img_face_illumination_1)
	print('img_face_occlusion_0, img_face_occlusion_1, img_face_occlusion_2', img_face_occlusion_0, img_face_occlusion_1, img_face_occlusion_2)
	# print img_height and img_width(min and max), and face_height and width (min and max)
	print('img width min & max ', img_width_min, img_width_max)
	print('img height min & max ', img_height_min, img_height_max)
	print('img_face width min & max ', face_width_min, face_width_max)
	print('img_face height min & max ', face_height_min, face_height_max)
	# i want to know the img height disturibution , so i use histogram, and i need to make some gap between the min and max of img_height
	# and by the max gap between the max and the min ,i want to make 120 gap,year
	for_iter = math.ceil(float(9108)/120)
	print('for_iter:', for_iter)
	createVar = locals()
	for ii in range(int(for_iter)):
		createVar['height_levle_gap_'+str(ii*120)] =0
	for ii in range(len(static_data.height)):
		for jj in range(int(for_iter)):
			if 120*(jj) <= static_data.height[ii] < 120*(jj+1):
				createVar['height_levle_gap_'+str(jj*120)] += 1
				break
	var_height_list = []
	for ii in range(int(for_iter)):
		var_height_list.append(createVar['height_levle_gap_'+str(ii*120)])
		if createVar['height_levle_gap_'+str(ii*120)]>0:
			print('height_levle_gap_'+str(ii*120),createVar['height_levle_gap_'+str(ii*120)])
	# ==========================================================
	# face_width min & max, and i will set gap between min & max
	# ==========================================================
	face_width_iter = math.ceil(float(976)/90)
	for ii in range(int(face_width_iter)):
		createVar['face_width_levle_gap_'+str(ii*90)] =0
	for ii in range(len(static_data.face_width)):
		for jj in range(int(face_width_iter)):
			if 90*(jj) <= static_data.face_width[ii] < 90*(jj+1):
				createVar['face_width_levle_gap_'+str(jj*90)] += 1
				break
	face_width_list = []
	for ii in range(int(face_width_iter)):
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
		face_height_list.append(createVar['face_height_levle_gap_'+str(ii*120)])
		print('face_height_levle_gap_'+str(ii*120),createVar['face_height_levle_gap_'+str(ii*120)])
	# ==================================================================
	# draw plot figure
	x = np.arange(for_iter)+1
	xx = np.arange(face_width_iter)+1
	xxx = np.arange(face_height_iter)+1
	# img_height_plot = plot.bar(x=x,height=var_height_list,width=0.85,facecolor='red', edgecolor = 'white', label='img_height_dist')
	#face_width_plot = plot.bar(x=xx+0.5,height=face_width_list,width=0.35,facecolor='green', edgecolor = 'white', label='face_width_dist')
	face_width_plot = plot.bar(x=xxx+0.5,height=face_height_list,width=0.35,facecolor='black', edgecolor = 'white', label='face_label_dist')
	# autolabel(img_height_plot)
	# autolabel(face_width_plot)
	plot.show()


# recursive function for the file of all readlines, maybe killed
def generate_label_file(label_line):
	if len(label_line) == 0:
		return
	image_file_name = label_line[0]
	print("image_file_name: %s"%{image_file_name})
	image_bbox = int(label_line[1])
	label_image_file = LABEL_DIR +'/'+ image_file_name.split('.')[-2].split('/')[-1]
	lab_file = open(label_image_file,'w')
	for ii in range(image_bbox):
		index_bbox = label_line[(ii + 1) + 1]
		lab_file.writelines(index_bbox)
	lab_file.close()
	return generate_label_file(label_line[2+image_bbox:])


# generate label txt file for each image
def convert_src_anno_label(split_file):
	mainSetFile = open(root_dir+ "mtfl/ImageSets/Main/" +str(split_file.split("/")[-1]), "w")
	with open(split_file, 'r') as label_file:
		while True:  # and len(faces)<10
			'''
			info = label_file.readline()
			if len(info) == 0:
				break;
			'''
			img_file_info = label_file.readline().split(' ')
			img_filename = img_file_info[1]
			img_filename = img_filename.replace('\\', '/')
			img_file = root_dir + "mtfl/JPEGImages/" + img_filename
			source_img = cv2.imread(img_file)
			fullImg = os.path.abspath(img_file) + '\n'
			mainSetFile.writelines(fullImg)
			labelFile = open(LABEL_DIR+'/'+fullImg.split("/")[-1].split(".")[0], "w")
			x1 = img_file_info[2]
			x2 = img_file_info[3]
			x3 = img_file_info[4]
			x4 = img_file_info[5]
			x5 = img_file_info[6]
			y1 = img_file_info[7]
			y2 = img_file_info[8]
			y3 = img_file_info[9]
			y4 = img_file_info[10]
			y5 = img_file_info[11]
			x =[x1, x2, x3, x4, x5];
			y =[y1, y2, y3, y4, y5];
			for ii in range(5):
				cv2.circle(source_img, (int(float(x[ii])), int(float(y[ii]))), 1, (0, 0, 225), -1)
			cv2.imwrite(Annotation_img_dir+"/"+str(img_file.split("/")[-1]), source_img)
			print (Annotation_img_dir+"/"+str(img_file.split("/")[-1]))
			gender = img_file_info[12]
			glass = img_file_info[14]
			headpose = img_file_info[15]
			landmark = x1 + " " + x2 + " " + x3 + " " + x4 \
						+ " " + x5 + " " + y1 + " " + y2 + " " \
						+ " " + y3 + " " + y4 + " " + y5 + " " + gender + " " + glass + " "+ headpose
			labelFile.write(landmark)
			labelFile.close()
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


def main():

	for sub in anno_mtfl_dir:
		dir = "../../dataset/facedata/mtfl/"+sub
		convert_src_anno_label(dir)
	# for file in wider_directory:
	# 	shuffle_file('../../dataset/facedata/wider_face/ImageSets/Main'+'/'+file+'.txt'
	# draw_histogram_base_data()
	# draw_histogram_specfic_range_base_data()


if __name__ == '__main__':
	main()



