from utils import util_data as utils
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import array_ops
import argparse
image_size = np.array([299,299],dtype=np.int32)
trainlabelfile = '../../dataset/actiondata/ucf101/ucf_label_101/ucf_101_label_train'
testlabelfile = '../../dataset/actiondata/ucf101/ucf_label_101/ucf_101_label_test'


def main():
	from tensorflow.python import pywrap_tensorflow
	checkpoint_path = '/home/resideo/workspace/action_recognition/resideo_action/models/' \
                      'action/20190330-171355/model-20190330-171355.ckpt-75271.index'
	reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
	var_to_shape_map = reader.get_variable_to_shape_map()
	for key in var_to_shape_map:
		print("tensor_name: ", key)


def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu_memory_fraction', type=float,
						help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
	parser.add_argument('--random_crop',
						help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
						'If the size of the images in the data directory is equal to image_size no cropping is performed',
						action='store_true', default=True)
	parser.add_argument('--random_flip',
						help='Performs random horizontal flipping of training images.', action='store_true', default=True)
	parser.add_argument('--random_rotate',
						help='Performs random rotations of training images.', action='store_true',  default=True)
	parser.add_argument('--use_fixed_image_standardization',
						help='Performs fixed standardization of images.', action='store_true', default=True)
	parser.add_argument('--keep_probability', type=float,
						help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
	return parser.parse_args(argv)


if __name__=="__main__":
	main()
