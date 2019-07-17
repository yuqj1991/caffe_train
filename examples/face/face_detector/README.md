# MobileNet-SSD
A caffe implementation of MobileNet-SSD detection network, with pretrained weights on VOC0712 and mAP=0.727.

Network|mAP|Download|Download
:---:|:---:|:---:|:---:
MobileNet-SSD|72.7|[train](https://drive.google.com/open?id=0B3gersZ2cHIxVFI1Rjd5aDgwOG8)|[deploy](https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc)

### Run
1. Download [SSD](https://github.com/weiliu89/caffe/tree/ssd) source code and compile (follow the SSD README).
2. Download the pretrained deploy weights from the link above.
3. Put all the files in SSD_HOME/examples/
4. Run demo.py to show the detection result.
5. You can run merge_bn.py to generate a no bn model, it will be much faster.

### Train your own dataset
1. Convert your own dataset to lmdb database (follow the SSD README), and create symlinks to current directory.
```
ln -s PATH_TO_YOUR_TRAIN_LMDB trainval_lmdb
ln -s PATH_TO_YOUR_TEST_LMDB test_lmdb
```
2. Create the labelmap.prototxt file and put it into current directory.
3. Use gen_model.sh to generate your own training prototxt.
4. Download the training weights from the link above, and run train.sh, after about 30000 iterations, the loss should be 1.5 - 2.5.
5. Run test.sh to evaluate the result.
6. Run merge_bn.py to generate your own no-bn caffemodel if necessary.
```
python merge_bn.py --model example/MobileNetSSD_deploy.prototxt --weights snapshot/mobilenet_iter_xxxxxx.caffemodel
```

### About some details
There are 2 primary differences between this model and [MobileNet-SSD on tensorflow](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md):
1. ReLU6 layer is replaced by ReLU.
2. For the conv11_mbox_prior layer, the anchors is [(0.2, 1.0), (0.2, 2.0), (0.2, 0.5)] vs tensorflow's [(0.1, 1.0), (0.2, 2.0), (0.2, 0.5)].

### Reproduce the result
I trained this model from a MobileNet classifier([caffemodel](https://drive.google.com/open?id=0B3gersZ2cHIxZi13UWF0OXBsZzA) and [prototxt](https://drive.google.com/open?id=0B3gersZ2cHIxWGEzbG5nSXpNQzA)) converted from [tensorflow](http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz). I first trained the model on MS-COCO and then fine-tuned on VOC0712. Without MS-COCO pretraining, it can only get mAP=0.68.

### Mobile Platform
You can run it on Android with my another project [rscnn](https://github.com/chuanqi305/rscnn).

### option
你能想到的别人也能想到，那到底如何能创造出自己的算法呢？
现在目标任务，实现人脸的对齐，6-10m的数据库来填充图片
task任务：
1、分类，1,是不是人，2,是不是部分人脸
2、边框回归;
3、坐标回归


一个网络要完成上述三个任务：
首先想想数据是怎样的
网络层如何进行设计
多层网络以基础网络来做，在每层网络上添加loss
分类loss：交叉熵损失代价函数;
边框box：边框回归损失函数
landmarks：点回归损失函数

有无人脸，有无半张人脸，

主体cnn：如何设计？还是需要预先了解输入的图片会使什么样的，才可以具体设计到底多少层？

先不用思考如何设计主体思路，还是应该需要拓宽数据库，这是因为我们这边需要从最开始，aflw and wilder face 这两部分，不和整理数据集，进行训练

deep-face的AI 网络框架：
1、数据层：主要包括什么：image/data/  label：person and whether partial face (iou <0.4):
数据集：widerface，and aflw landmarks
还得要剔除到wideface上很多人脸过小的数据，因为过小的人脸会影响到数据（为什么不能剔除人脸呢？过小的人脸像素不够充足，不能影响，那么需要该如何去测试，像素的图片大小，那么在对比像素的面积呢）
需要做的工作：生成部分人脸，生成的标准：准备生成多少张
aflw：  做网络人脸的精修，包括landmarks的标定，
那么网络的架构，敬请期待明日：
