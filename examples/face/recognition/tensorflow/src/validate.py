import os
import sys
import math

caffe_feature = []
movidious_feature_v1 = []
movidius_feature_v2 = []
tensorflow_feature = []

tensorflow_feature_dir = "./faceDataTensorflow/"
caffe_feature_dir = "./faceDataCaffe/"
movidius_feature_v2_dir = "./faceDataMvdiousV2.005/"

def computeDistance(feature_a, feature_b):
	assert len(feature_a) == len(feature_b)
	top = 0.0
	bottom_left = 0.0
	bottom_right = 0.0
	for i in range(len(feature_a)):
		top += feature_a[i] * feature_b[i]
		bottom_left += feature_a[i] * feature_a[i]
		bottom_right += feature_b[i] * feature_b[i]
	cosValue = float(top / (math.sqrt(bottom_left)*math.sqrt(bottom_right)))
	return cosValue


def valdiateSamePerson():
	for file_feature in os.listdir(tensorflow_feature_dir):
		# tensorflow feature
		file_path_tensorflow = tensorflow_feature_dir + file_feature
		data_file_ = open(file_path_tensorflow, 'r')
		content_tensorflow_lines = data_file_.read()
		feature_tensorflow_list = []
		for i in content_tensorflow_lines.split():
			value = float(i)
			feature_tensorflow_list.append(value)
		data_file_.close()
		# caffe feature
		file_path_caffe = caffe_feature_dir + file_feature
		data_file_caffe_ = open(file_path_caffe, 'r')
		content_caffe_lines = data_file_caffe_.read()
		feature_caffe_list = []
		for i in content_caffe_lines.split():
			value = float(i)
			#print(value)
			feature_caffe_list.append(value)
		data_file_caffe_.close()
		# movidius_v2.0 feature
		file_path_movidus_v2 = movidius_feature_v2_dir + file_feature
		data_file_movidus_v2_ = open(file_path_movidus_v2, 'r')
		content_movidus_v2_lines = data_file_movidus_v2_.read()
		feature_movidus_v2_list = []
		for i in content_movidus_v2_lines.split():
			value = float(i)
			feature_movidus_v2_list.append(value)
		data_file_movidus_v2_.close()
		# computeDistance caffe & tensorflow
		cosValue_tensorflow_caffe = computeDistance(feature_tensorflow_list, feature_caffe_list)
		print("valdiateSamePerson caffe & tensorflow cosvalue distance: ", cosValue_tensorflow_caffe)
		# computeDistance movidious_v2 & tensorflow
		cosValue_tensorflow_movidious_v2 = computeDistance(feature_tensorflow_list, feature_movidus_v2_list)
		print("valdiateSamePerson movidious_v2 & tensorflow cosvalue distance: ", cosValue_tensorflow_movidious_v2)


def valdiatedifferentPerson():
	for file_feature in os.listdir(tensorflow_feature_dir):
		# tensorflow feature
		file_path_tensorflow = tensorflow_feature_dir + file_feature
		data_file_ = open(file_path_tensorflow, 'r')
		content_tensorflow_lines = data_file_.read()
		feature_tensorflow_list = []
		for i in content_tensorflow_lines.split():
			value = float(i)
			feature_tensorflow_list.append(value)
		data_file_.close()
		# caffe feature for other persons
		for caffe_file in os.listdir(caffe_feature_dir):
			if file_feature != caffe_file:
				file_path_caffe = caffe_feature_dir + caffe_file
				data_file_caffe_ = open(file_path_caffe, 'r')
				content_caffe_lines = data_file_caffe_.read()
				feature_caffe_list = []
				for i in content_caffe_lines.split():
					value = float(i)
					#print(value)
					feature_caffe_list.append(value)
				data_file_caffe_.close()
				# computeDistance caffe & tensorflow
				cosValue_tensorflow_caffe = computeDistance(feature_tensorflow_list, feature_caffe_list)
				print("valdiatedifferentPerson caffe & tensorflow at %s and %s is cosvalue distance: %f"%(caffe_file, file_feature, cosValue_tensorflow_caffe))
			else:
				continue
		# movidius_v2.0 feature
		'''
		for movidious_file_ in os.listdir(movidius_feature_v2_dir):
			if movidious_file_ != file_feature:
				file_path_movidus_v2 = movidius_feature_v2_dir + movidious_file_
				data_file_movidus_v2_ = open(file_path_movidus_v2, 'r')
				content_movidus_v2_lines = data_file_movidus_v2_.read()
				feature_movidus_v2_list = []
				for i in content_movidus_v2_lines.split():
					value = float(i)
					feature_movidus_v2_list.append(value)
				data_file_movidus_v2_.close()
				# computeDistance movidious_v2 & tensorflow
				cosValue_tensorflow_movidious_v2 = computeDistance(feature_tensorflow_list, feature_movidus_v2_list)
				print("valdiatedifferentPerson movidious_v2 & tensorflow at %s and %s is cosvalue distance: %f"%(movidious_file_, file_feature, cosValue_tensorflow_movidious_v2))
			else:
				continue
		'''

if __name__=='__main__':
	#valdiateSamePerson()
	valdiatedifferentPerson()

