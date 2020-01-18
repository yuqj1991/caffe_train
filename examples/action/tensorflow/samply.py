import tensorflow as tf
import cv2
filename = './data/image_0002.jpg'
file_contents = tf.read_file(filename)
image = tf.image.decode_image(file_contents, channels=3)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	img =sess.run((image))
	print(img.shape)
	print(img[1::])

	cv2.imshow('test', img)
	cv2.waitKey(0)