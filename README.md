## caffe train face licenseplate reID action ocr

---

2020.07.11. Updates: optimizing the CUDA memory to improve the batch_size in one batch at the limited Cuda memory of 16GitaBytes.

---

## caffe implement method
| Method | orginal performance| caffe performance | 
|:--------:| :--------:| :---------:| 
|focal loss|            |            | 
|data_anchor_sample|        |        |
|cos_loss|              |            |
|Yolov3  |              |            |
|triplet_loss|          |            |

---

## centernet face + nms version  

| Method | Easy | Medium | Hard|
|:--------:| :--------:| :---------:| :------:|
| ours(one scale(640x640) + nms)| 0.8388 | 0.8223   | 0.7246 |
| ours(one scale(640x640))| 0.8326 | 0.8147   | 0.6955 |
| ours(one scale(800x800) + nms)| 0.8267 | 0.8194   | 0.7672 |
| original | 0.922 | 0.911 | 0.782 |

---

## example val accuray  
| Method | accuray performance|  
|:--------:| :--------:|  
|mobilenet-v2 face recognition|99.42%  |
|face landmarks + face attributes gender |(99.4%)+ bool glasses(99.5%)|  
|face head angle|(not evaluated)|  
|car license plate detect|(mAP91%)|  
|car recognition| (96%, only support anhui car + blue car style)|  

## New proposed method CenterGridSoftmax + nms method by using anchor  
| Method | Easy | Medium | Hard|
|:--------:| :--------:| :---------:| :------:|
| ours(one scale)| 0.8636 | 0.8489   | 0.7425 |

---

## will add caffe version efficientDet, to be continued
we had add the net prototxt, not training


## python Caffe API
import caffe  
from caffe import layers as L  
from caffe import params as P  
L.Data( source=lmdb,backend=P.Data.LMDB,batch_size=batch_size, ntop=2,transform_param=dict(crop_size=227,mean_value=[104, 117, 123],mirror=True))  
L.HDF5Data(hdf5_data_param={'source': './training_data_paths.txt','batch_size': 64},include={'phase': caffe.TRAIN})  
L.ImageData(source=list_path,batch_size=batch_size,new_width=48,new_height=48,ntop=2,ransform_param=dict(crop_size=40,mirror=True))  
L.Convolution(bottom, kernel_size=ks, stride=stride,num_output=nout, pad=pad, group=group)  
L.LRN(bottom,local_size=5, alpha=1e-4, beta=0.75)  
L.ReLU(bottom, in_place=True)  
L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)  
L.InnerProduct(bottom, num_output=nout)  
L.Dropout(bottom, in_place=True)  
L.SoftmaxWithLoss(bottom, label)  

查看某一层的宽，高，shape，可以用net['layer_name'].blobs[].data.shape[]

---

## 数据增强方法

假设原图输入是一张640*480的图片，这里由于版面问题我放缩了图片尺寸并且没做mean subtract，由于最后会有resize参数导致输出的图片都会resize到300x300，但是主要看的是增强的效果，SSD中的数据增强的顺序是：

DistortImage: 这个主要是修改图片的brightness，contrast，saturation，hue，reordering channels，并没改变标签bbox

ExpandImage: 这个主要是将DistortImage的图片用像素0进行扩展，标签bbox此时肯定会改变，就重新以黑边的左上角为原点计算[0,1]的bbox的左上角和右下角两个点坐标。

BatchSampler: 由于这里选错图了，BatchSampler必须要有GT的存在才会生效，由于我做的是人的检测所以图中没人就不会生成sampled_bboxes，后面修改例子。sampled_bboxes的值是随机在[0, 1]上生成的bbox，并且和某个gt_bboxes的IOU在[min, max]之间。由于proto中配的max_sample都是为1，所以每个batch_sampler可能会有1个sampled_bbox，随机取一个sampled bbox并且裁剪图片和标签。标签裁剪也很好理解首先要通过ProjectBBox将原坐标系标签投影到裁剪后图片的新坐标系的坐标，然后再ClipBBox到[0,1]之间。

Resize：放缩到300x300，最后将图片放缩到300x300，标签框也是线性放缩坐标而已。

Crop：原本data_transformer还会crop的，这个参数是配在prototxt中，默认是原图 所以就和没crop一样。如果要crop的话标签也是会和之前BatchSampler那样处理。

data_anchor_sampler: 在一张图像中随机抽取一张sface
