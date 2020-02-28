caffe train face licenseplate reID action ocr  
focal loss layer implemented by caffe  
cosin face loss layer implemented by caffe  
data augement for pymraidBox implemented by caffe  
yolo layer & reorg layer for darknet implemented by caffe  
centernet implemented by caffe  
facenet tripletloss by caffe

LFFD insight: 
anchor box generate progress:
anchor size is the receptive size : in terms of 640x640 placed in the folder(examples/face/detector/prototxt/Full_640x640/train_v2.prototxt)
feature_map_size_list = {160, 160, 80, 80, 40, 20, 20, 20}
specialiled receptive_size_list = {15, 20, 40, 70, 110, 250, 400, 560}
bbox_small_list = {10, 15, 20, 40, 110, 250, 400}
bbox_large_list = {15, 20, 40, 70, 110, 250, 400, 560}
bbox_gray_small_scale_list = 0.9 * bbox_small_list
bbox_gray_large_scale_list = 1.1 * bbox_large_list
receptive_field_center_stride_list = {4, 4, 8, 8, 16, 32, 32, 32}
receptive_field_center_start_list = {3, 3, 7, 7, 15, 31, 31, 31}
num_output_scales = 8
if given a fixed gt_box(xmin, xmax, ymin, ymax):
	总过有8层输出，分别为box 边框回归，和类别分类， 对于每一层知道featuremap大小，以及该层的中心起始位置以及该层对应的anchor大小，可以确定出来这一层的anchor位置，然后
	求出anchor中心落在gt_box里面，同时还需要满足，1）相应的gt_box size与anchor size是匹配的；2）gt_box size 落在gray_scale_bbox相应的范围内的，gt_box是要被忽略吗？存疑
	3）多个anchor匹配到同一个gt_box 这其他anchor需要被忽略
最后困难样本发觉：1:10 

