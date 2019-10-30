import os
import tensorflow as tf
import numpy as np
import cv2


Resized_size_width = 320
Resized_size_height = 320
MAX_EPOCHS = 20
NUM_BATCHES = 30

BATCH_IMAGES = 300

sub_batch_images = 9

imagefilepath = "/home/deepano/workspace/caffe_train/examples/face/detector/scripts/wider_train.txt"

images_path_holder = tf.placeholder(dtype=tf.string, shape=(None, ), name = "images_path")
labels_holder = tf.placeholder(dtype=tf.string, shape=(None, ), name="labels")
batch_size_holder = tf.placeholder(dtype=tf.int64, name="batch_size")


def get_list_from_label_file(image_label_file_, batch_size):
	image_list = []
	label_list = []
	i = 0
	with open(image_label_file_, 'r') as anno_file_:
		for contentline in anno_file_.readlines():
			curLine=contentline.strip().split(' ')
			image_list.append(curLine[0])
			label_list.append(curLine[1])
	anno_file_.close()
	print("length: ",len(image_list))
	return image_list, label_list


def sample_select(image_list_, batch_size):
	image_list = []
	label_list = []
	for i in range(batch_size):
		image_list.append(image_list_[i])
		label_list.append(i)
	return image_list, label_list



def load_batch_data(filename, label):
	img_data = []
	label_data = []
	print(filename)
	image_ = cv2.imread(filename)
	image_ = cv2.resize(image_, (Resized_size_width, Resized_size_height))
	img_data.append(image_)
	return image_, label
	
def _parse_image(filename, label):
	"""
	read and pre-process image
	"""
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string)

	image_converted = tf.cast(image_decoded, tf.float32)
	resized_image = tf.image.resize_images(image_converted, [320, 320], method=0)
	#image_scaled = tf.divide(tf.subtract(image_converted, 3 / 2), 3)
	return resized_image, label


def modelnet(inputs):
	with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
		kernel_filter = tf.get_variable(name = "weight", shape = [3, 3, 3, 32], trainable=True, initializer=tf.contrib.layers.xavier_initializer(),  dtype=tf.float32)
		biase_add = tf.get_variable(name="biase", shape=[32],  dtype=tf.float32, trainable = True)
	conv_1= tf.nn.conv2d(input = inputs, filter=kernel_filter, strides = [1, 2, 2, 1], padding='SAME')
	conv_1_biase = tf.nn.bias_add(conv_1, biase_add)
	output = tf.nn.relu(conv_1_biase)
	return output


image_list_batch, label_list_batch = get_list_from_label_file(imagefilepath, sub_batch_images)

#image_list_batch_, label_list_batch_ = sample_select(image_list_batch, 5)
'''
filenames = tf.constant(['/home/deepano/workspace/dataset/facedata/wider_face/annoImg/2--Demonstration_2_Demonstration_Demonstration_Or_Protest_2_798.jpg','/home/deepano/workspace/dataset/facedata/wider_face/annoImg/19--Couple_19_Couple_Couple_19_366.jpg','/home/deepano/workspace/dataset/facedata/wider_face/annoImg/50--Celebration_Or_Party_50_Celebration_Or_Party_houseparty_50_780.jpg','/home/deepano/workspace/dataset/facedata/wider_face/annoImg/2--Demonstration_2_Demonstration_Demonstration_Or_Protest_2_798.jpg','/home/deepano/workspace/dataset/facedata/wider_face/annoImg/19--Couple_19_Couple_Couple_19_366.jpg','/home/deepano/workspace/dataset/facedata/wider_face/annoImg/50--Celebration_Or_Party_50_Celebration_Or_Party_houseparty_50_780.jpg'])
labels = tf.constant([1,2,3, 4, 5,5])
'''

dataset = tf.data.Dataset.from_tensor_slices((images_path_holder, labels_holder))

dataset = dataset.map(_parse_image)

dataset = dataset.shuffle(20).batch(batch_size_holder).repeat()


print("dataset.output_shapes",dataset.output_shapes)

iterator = dataset.make_initializable_iterator()


images_, labels_ = iterator.get_next()
output = modelnet(images_)
sess = tf.Session()



sess.run(tf.global_variables_initializer())

nrof_batches = int(np.ceil(BATCH_IMAGES / sub_batch_images))

for i in range(MAX_EPOCHS):
	for step in range(BATCH_IMAGES):
		sess.run(iterator.initializer, feed_dict={images_path_holder: image_list_batch,labels_holder: label_list_batch, batch_size_holder: sub_batch_images})
		output_ = sess.run(output)
		for i in range(nrof_batches):
			output_ = sess.run(output)
		sess.run(iterator.initializer, feed_dict={images_path_holder: image_list_batch,labels_holder: label_list_batch, batch_size_holder: 5})
		for i in range(nrof_batches):
			output_12 = sess.run(output)
			print("*******: %d", i)
	print("epoch: %d", i)



def 



