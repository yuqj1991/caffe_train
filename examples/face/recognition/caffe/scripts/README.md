FaceNet
=======

Data Preprocess
---
>> vggface2 dataset

>>(1)人脸对齐  
>>使用vggface_align.py脚本进行人脸对齐，对齐方法：读取loose_bb_train.csv标注文件中人脸框，人脸框4个方向各外扩20%，抠取人脸，然后把人脸框的最小边缩放到128（宽高等比例缩放）。

>>对齐后的人脸如下:  

>>(2)人脸列表  
>>利用face_labels_gen.py脚本生成face label

>>(3)生成LMDB  
>>利用create_vggface2.sh脚本生成caffe训练时用的lmdb数据。

Train
---
>>./build/tools/caffe train --solver examples/face_recog/inception_resnet_v2_tiny_solver.prototxt  --gpu 0,1

Test
---
>>(1)LFW数据集测评：  
>>利用face_similarity.py脚本生成LFW人脸对的余弦相似度得分。  
>>python face_similarity.py   --image_dir --image_dir ../../../../../../dataset/facedata/lfw/lfw_align_160/  --pair_file face_list_lfw.txt  --result_file  facenet_vggface2_inception_resnet_v2_lfw.txt  --network deploy.prototxt  --weights deploy.caffemodel  

>>(2)LFW性能分析：  
>>利用roc_curve.py脚本生成结果。  
>> python roc_curve.py gt_face_list_lfw.txt  facenet_vggface2_inception_resnet_v2_lfw.txt
