import os, sys
import numpy as np
import cv2
from PIL import Image
import argparse
import random
from scipy import misc
import math
import re
import tensorflow as tf
from tensorflow.python.platform import gfile
# my idea is input five images into the cnn net and get the 512-dimension feature, and then the input data will be
# feed into the lstm ,time_step is 5
# training detail:include training parameter,like data size
# data moudle: get sequence image data and related label
# shuffle origin batch, and reset new batch.

RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
classind = '/home/resideo/workspace/dataset/actiondata/ucf101/train_test_indices/classInd.txt'
model = ['train', 'test']


class rawdatasource():
	"Stores the paths to images for a given class"
	def __init__(self, image_path, label):
		self.label = label
		self.image_path = image_path


def get_model_filenames(model_dir):
	files = os.listdir(model_dir)
	meta_files = [s for s in files if s.endswith('.meta')]
	if len(meta_files)==0:
		raise ValueError('No meta file found in the model directory (%s)' % model_dir)
	elif len(meta_files)>1:
		raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
	meta_file = meta_files[0]
	ckpt = tf.train.get_checkpoint_state(model_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
		return meta_file, ckpt_file
	meta_files = [s for s in files if '.ckpt' in s]
	max_step = -1
	for f in files:
		step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
		if step_str is not None and len(step_str.groups())>=2:
			step = int(step_str.groups()[1])
			if step > max_step:
				max_step = step
				ckpt_file = step_str.groups()[0]
	return meta_file, ckpt_file


def load_model(model, input_map=None):
	# Check if the model is a model directory (containing a metagraph and a checkpoint file)
	# or if it is a protobuf file with a frozen graph
	model_exp = os.path.expanduser(model)
	if (os.path.isfile(model_exp)):
		print('Model filename: %s' % model_exp)
		with gfile.FastGFile(model_exp, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			tf.import_graph_def(graph_def, input_map=input_map, name='')
	else:
		print('Model directory: %s' % model_exp)
		meta_file, ckpt_file = get_model_filenames(model_exp)
		print('Metagraph file: %s' % meta_file)
		print('Checkpoint file: %s' % ckpt_file)
		saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
		saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_image_paths_and_labels(trainLableFile):
	image_paths_flat = []
	labels_flat = []
	with open(trainLableFile) as trainfile:
		while True:
			readInfo = trainfile.readline()
			if readInfo =='':
				break
			image_paths_flat.append(readInfo.split(' ')[0])
			labels_flat.append(readInfo.split(' ')[-1].replace('\n','').replace('\r',''))
	print("read %s labelfile end"%trainLableFile)
	return image_paths_flat, labels_flat


def get_map_label_indices(classfile):
	label_map_indices = {}
	with open(classfile) as file:
		while True:
			lineInfo = file.readline()
			class_ind =lineInfo.split(' ')[0]
			if class_ind =='':
				break
			ind = str(int(class_ind)-1)
			classinfo = lineInfo.split(' ')[1].replace('\n','').replace('\r','')
			if classinfo=='':
				break
			label_map_indices[classinfo] = ind
		file.close()
	return label_map_indices


def shuffle_video_file(video_file_directory,label_category_list, batch_size):
	'''
	shuffle video file name and get the training file list and random
	:param video_file_directory: training file list set
	:param label_category_list:
	:param batch_size: training batch size
	:return:
	'''
	assert len(video_file_directory) == len(label_category_list)
	assert len(video_file_directory) >= batch_size
	np.random.shuffle(video_file_directory)
	image_label_batch_list = np.random.randint(0,len(video_file_directory),batch_size)
	image_batch_file = []
	label_batch = []
	for index in range(batch_size):
		image_batch_file.append(video_file_directory[image_label_batch_list[index]])
		label_batch.append(label_category_list[image_label_batch_list[index]])
	return image_batch_file, label_batch


# how to define label file, every frame map to one label
# return image_batch_data, and label_file, gai ruhelianghuazhege zhege label.
def load_video_data(image_resize, image_batch_file_path,image_save_folder,label_save_folder,step_frame,model):
	resize_img_h = image_resize[0]
	resize_img_w = image_resize[1]
	if not os.path.isdir(image_save_folder):
		os.mkdir(image_save_folder)
	if not os.path.isdir(label_save_folder):
		os.mkdir(label_save_folder)
	label_file = open(label_save_folder+'/'+'ucf_101_label_'+model,'w')
	with open(image_batch_file_path, 'r') as train_label_file:
		while True:
			lineInfo = train_label_file.readline()
			test_file = lineInfo.split(' ')[0]
			if test_file =='':
				break
			video_file_name = '../../dataset/actiondata/ucf101/ucf_origin_101/'+test_file
			sub_video_directory = test_file.split('/')[0]
			cat_label_name=str(int(lineInfo.split(' ')[1])-1)
			image_file = test_file.split('/')[-1].split('.avi')[0]
			if not os.path.isdir(image_save_folder+'/'+sub_video_directory):
				os.mkdir(image_save_folder+'/'+sub_video_directory)
			cap = cv2.VideoCapture(video_file_name)
			index_frame = 0
			while True:
				ret, frame = cap.read()
				if frame is None:
					break
				cvt_frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
				resize_image = cv2.resize(cvt_frame, (resize_img_w, resize_img_h))
				if index_frame%step_frame == 0:
					cv2.imwrite(image_save_folder+'/'+sub_video_directory+'/'+image_file+'_'+str(index_frame) +
							'.jpg', resize_image)
					image_full_label_path = image_save_folder+'/'+sub_video_directory+'/'+image_file+'_' + \
									str(index_frame)+'.jpg'+' '+cat_label_name+'\n'
					print(image_full_label_path)
					label_file.writelines(image_full_label_path)
				index_frame += 1
	label_file.close()
	return


def load_video_test_data(image_resize, image_batch_file_path,image_save_folder,label_save_folder,step_frame,model, labelmap):
	resize_img_h = image_resize[0]
	resize_img_w = image_resize[1]
	if not os.path.isdir(image_save_folder):
		os.mkdir(image_save_folder)
	if not os.path.isdir(label_save_folder):
		os.mkdir(label_save_folder)
	label_file = open(label_save_folder+'/'+'ucf_101_label_'+model,'w')
	with open(image_batch_file_path, 'r') as train_label_file:
		while True:
			test_file = train_label_file.readline().split('\n')[0].replace('\r','')
			print(test_file)
			video_file_name = '../../dataset/actiondata/ucf101/ucf_origin_101/'+test_file
			print(video_file_name)
			sub_video_directory = test_file.split('/')[0]
			cat_label_name=test_file.split('/')[0]
			if cat_label_name in labelmap:
				label_ind = labelmap[cat_label_name]
			image_file = test_file.split('/')[-1].split('.avi')[0]
			if not os.path.isdir(image_save_folder+'/'+sub_video_directory):
				os.mkdir(image_save_folder+'/'+sub_video_directory)
			cap = cv2.VideoCapture(video_file_name)
			index_frame = 0
			while True:
				ret, frame = cap.read()
				if frame is None:
					break
				cvt_frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
				resize_image = cv2.resize(cvt_frame, (resize_img_w, resize_img_h))
				if index_frame%step_frame == 0:
					cv2.imwrite(image_save_folder+'/'+sub_video_directory+'/'+image_file+'_'+str(index_frame) +
							'.jpg', resize_image)
					image_full_label_path = image_save_folder+'/'+sub_video_directory+'/'+image_file+'_' + \
									str(index_frame)+'.jpg'+' '+label_ind+'\n'
					print(image_full_label_path)
					label_file.writelines(image_full_label_path)
				index_frame += 1
	label_file.close()
	return


def shuffle_samples(image_paths, labels):
	shuffle_list = list(zip(image_paths, labels))
	random.shuffle(shuffle_list)
	image_paths_shuff, labels_shuff = zip(*shuffle_list)
	return image_paths_shuff, labels_shuff


def random_rotate_image(image):
	rotate_angle = np.random.uniform(low=-10.0, high=10.0)
	return misc.rotate(image, rotate_angle)


def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
	if mode=='SPLIT_CLASSES':
		nrof_classes = len(dataset)
		class_indices = np.arange(nrof_classes)
		np.random.shuffle(class_indices)
		split = int(round(nrof_classes*(1-split_ratio)))
		train_set = [dataset[i] for i in class_indices[0:split]]
		test_set = [dataset[i] for i in class_indices[split:-1]]
	elif mode == 'SPLIT_IMAGES':
		train_set = []
		test_set = []
		for cls in dataset:
			paths = cls.image_paths
			np.random.shuffle(paths)
			nrof_images_in_class = len(paths)
			split = int(math.floor(nrof_images_in_class*(1-split_ratio)))
			if split == nrof_images_in_class:
				split = nrof_images_in_class-1
			if split >= min_nrof_images_per_class and nrof_images_in_class-split>=1:
				train_set.append(rawdatasource(cls.name, paths[:split]))
				test_set.append(rawdatasource(cls.name, paths[split:]))
	else:
		raise ValueError('Invalid train/test split mode "%s"' % mode)
	return train_set, test_set


def get_image(image_path):
	content = tf.read_file(image_path)
	tf_image = tf.image.decode_jpeg(content, channels=3)
	return tf_image


def create_input_pipeline(input_queue, image_size, nrof_precess_threads, batch_size_placeholder):
	images_and_labels_list = []
	for _ in range(nrof_precess_threads):
		filenames, label, control = input_queue.dequeue()
		images = []
		for filename in tf.unstack(filenames):
			file_contents = tf.read_file(filename)
			image = tf.image.decode_image(file_contents, channels=3)
			if 0:
				print('get_control_flag(control[0], RANDOM_ROTATE): ', get_control_flag(control[0], RANDOM_ROTATE))
				print('get_control_flag(control[0], RANDOM_CROP): ', get_control_flag(control[0], RANDOM_CROP))
				print('get_control_flag(control[0], RANDOM_FLIP): ', get_control_flag(control[0], RANDOM_FLIP))
				print('get_control_flag(control[0], FIXED_STANDARDIZATION): ',get_control_flag(control[0], FIXED_STANDARDIZATION))
				print('get_control_flag(control[0], FLIP): ', get_control_flag(control[0], FLIP))
			image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),
							lambda: tf.py_func(random_rotate_image, [image], tf.uint8),
							lambda: tf.identity(image))
			image = tf.cond(get_control_flag(control[0], RANDOM_CROP),
			                lambda: tf.random_crop(image, image_size),
			                lambda: tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
			image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),
							lambda: tf.image.random_flip_left_right(image),
							lambda: tf.identity(image))
			image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),
							lambda: (tf.cast(image, tf.float32) - 127.5) / 128.0,
							lambda: tf.image.per_image_standardization(image))
			image = tf.cond(get_control_flag(control[0], FLIP),
							lambda: tf.image.flip_left_right(image),
							lambda: tf.identity(image))
			image.set_shape((image_size[0], image_size[1], 3))
			images.append(image)
		images_and_labels_list.append([images, label])
	image_batch, label_batch = tf.train.batch_join(images_and_labels_list, batch_size=batch_size_placeholder,
													shapes=[(image_size[0], image_size[1], 3), ()], enqueue_many=True,
													capacity=4 * nrof_precess_threads * 64,
													allow_smaller_final_batch=True)
	return image_batch, label_batch


def _int64_feature(value): # genterate intergers
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):# genterate bytes
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set,name):
    '''
    将数据填入到tf.train.Example的协议缓冲区（protocol buffer)中，将协议缓冲区序列
    化为一个字符串，通过tf.python_io.TFRecordWriter写入TFRecords文件
    '''
    images=data_set.images
    labels=data_set.labels
    num_examples=data_set.num_examples
    if images.shape[0]!=num_examples:
        raise ValueError ('Imagessize %d does not match label size %d.'\
                          %(images.shape[0],num_examples))
    rows=images.shape[1]    #image height
    cols=images.shape[2]    #image width
    depth=images.shape[3]   #channels

    filename = os.path.join(name, name + '.tfrecords')
    print('Writing',filename)
    writer=tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw=images[index].tostring()  #cast imagedata to string
        #写入协议缓冲区，height、width、depth、label编码成int 64类型，image——raw编码成二进制
        example=tf.train.Example(features=tf.train.Features(feature={
                'height':_int64_feature(rows),
                'width':_int64_feature(cols),
                'depth':_int64_feature(depth),
                'label':_int64_feature(int(labels[index])),
                'image_raw':_bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())       #序列化字符串
    writer.close()


def read_and_decode(filename_queue, image_shape):     #输入文件名队列
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    #解析一个example,如果需要解析多个样例，使用parse_example函数
    features=tf.parse_single_example(
            serialized_example,
            #必须写明feature里面的key的名称
            features={
            #TensorFlow提供两种不同的属性解析方法，一种方法是tf.FixedLenFeature,
            #这种方法解析的结果为一个Tensor。另一个方法是tf.VarLenFeature,
            #这种方法得到的解析结果为SparseTensor,用于处理稀疏数据。
            #这里解析数据的格式需要和上面程序写入数据的格式一致
                    'image_raw':tf.FixedLenFeature([],tf.string),#图片是string类型
                      'label':tf.FixedLenFeature([],tf.int64),  #标记是int64类型
                      })
    #对于BytesList,要重新进行编码，把string类型的0维Tensor变成uint8类型的一维Tensor
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([image_shape])
    #tensor("input/DecodeRaw:0",shape=(784,),dtype=uint8)
    #image张量的形状为：tensor("input/sub:0",shape=(784,),dtype=float32) , 可做可不做
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    #把标记从uint8类型转换为int32类性
    #label张量的形状为tensor（“input/cast_1:0",shape=(),dtype=int32)
    label = tf.cast(features['label'], tf.int32)
    return image, label


def inputs(train,batch_size,num_epochs, trainRecord, validationRecord):
    #输入参数：
    #train：选择输入训练数据/验证数据
    #batch_size:训练的每一批有多少个样本
    #num_epochs:过几遍数据，设置为0/None表示永远训练下去
    #返回结果：
	#A tuple (images,labels)
    #*images:类型为float，形状为【batch_size,mnist.IMAGE_PIXELS],范围【-0.5，0.5】。
    #*label:类型为int32，形状为【batch_size],范围【0，mnist.NUM_CLASSES]
    #注意tf.train.QueueRunner必须用tf.train.start_queue_runners()来启动线程

    if not num_epochs:num_epochs=None
    #获取文件路径，即./MNIST_data/train.tfrecords,./MNIST_data/validation.records
    filename=os.path.join('../data/',trainRecord if train else validationRecord)
    with tf.name_scope('input'):
        #tf.train.string_input_producer返回一个QueueRunner,里面有一个FIFOQueue
        filename_queue=tf.train.string_input_producer(#如果样本量很大，可以分成若干文件，把文件名列表传入
                [filename],num_epochs=num_epochs)
        image,label=read_and_decode(filename_queue)
        #随机化example,并把它们整合成batch_size大小
        #tf.train.shuffle_batch生成了RandomShuffleQueue,并开启两个线程
        images,sparse_labels=tf.train.shuffle_batch(
                [image,label],batch_size=batch_size,num_threads=2,
                capacity=1000+3*batch_size,
                min_after_dequeue=1000) #留下一部分队列，来保证每次有足够的数据做随机打乱
        return images,sparse_labels


def get_learing_rate():
   return 0.001


def _add_loss_summaries(total_loss):
	"""Add summaries for losses.

	Generates moving average for all losses and associated summaries for
	visualizing the performance of the network.

	Args:
	  total_loss: Total loss from loss().
	Returns:
	  loss_averages_op: op for generating moving averages of losses.
	"""
	# Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summmary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss
		# as the original loss name.
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))

	return loss_averages_op


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
		log_histograms=True):
	# Generate moving averages of all losses and associated summaries.
	loss_averages_op = _add_loss_summaries(total_loss)

	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		if optimizer == 'ADAGRAD':
			opt = tf.train.AdagradOptimizer(learning_rate)
		elif optimizer == 'ADADELTA':
			opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
		elif optimizer == 'ADAM':
			opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
		elif optimizer == 'RMSPROP':
			opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
		elif optimizer == 'MOM':
			opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
		else:
			raise ValueError('Invalid optimization algorithm')
		grads = opt.compute_gradients(total_loss,update_gradient_vars)
	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Add histograms for trainable variables.
	if log_histograms:
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)

	# Add histograms for gradients.
	if log_histograms:
		for grad, var in grads:
			if grad is not None:
				tf.summary.histogram(var.op.name + '/gradients', grad)

	# Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(
		moving_average_decay, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op


def get_control_flag(control, filed):
	return np.equal(np.mod(np.floor_divide(control, filed), 2), 1)


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                if par[1]=='-':
                    lr = -1
                else:
                    lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


def config_video_parameter(argv):
	parser = argparse.ArgumentParser(description="this script to configure image split")
	parser.add_argument("-resize_image_h", default=299, type=int,
						action="store", help="this is the resize_image_height" )
	parser.add_argument("-resize_image_w", default=299, type=int,
						action="store", help="this is the resize_image_width")
	parser.add_argument("-train_file_list", type=str,
						default="/home/resideo/workspace/dataset/actiondata/ucf101",
						action="store", help="type the source video sequence folder")
	parser.add_argument("-image_save_folder", type=str,
						default="/home/resideo/workspace/dataset/actiondata/ucf101/ucf_split_101")
	return parser.parse_args(argv)


def main(args):
	resize_img_h = args.resize_image_h
	resize_img_w = args.resize_image_w
	source_video_folder = args.train_file_list
	img_save_folder = args.image_save_folder
	resize_array = np.array([resize_img_h,resize_img_w])
	source_video_train = source_video_folder+'/train_test_indices/'+ 'trainlist01.txt'
	model_train = 'train'
	load_video_data(resize_array, source_video_train, img_save_folder,
	                "../../dataset/actiondata/ucf101/ucf_label_101", 10, model_train)
	labelmap = get_map_label_indices(classind)
	source_video_test = source_video_folder+'/train_test_indices/'+ 'testlist01.txt'
	model_test = 'test'
	load_video_test_data(resize_array, source_video_test, img_save_folder,
				"../../dataset/actiondata/ucf101/ucf_label_101", 10, model_test, labelmap)


if __name__ == '__main__':
	main(config_video_parameter(sys.argv[1:]))
