# 训练部分
下载：<br/>
舌头检测测试样本[train-samples-st.rar](http://www.zifuture.com/fs/12.github/mtcnn/train-samples-st.rar)<br/>
<br/>

## 无关键点的步骤：<br/>
1.准备好训练的样本图片放到samples文件夹<br/>
2.准备好对应的label.txt，格式是<br/>
   samples/filename.jpg xmin ymin xmax ymax<br/>
3.执行callpy-gen-data12.bat生成样本数据<br/>
4.执行callpy-12.bat生成训练需要用到的train-label.txt<br/>
5.执行make-lmdb-12.bat生成lmdb数据库<br/>
6.执行train-12.bat开始训练<br/>
7.相对应的其他24、48网络也类似就好了<br/>

<br/>
对于训练caffe.exe、转换数据集convert_imageset.exe（因为用到了--backend=mtcnn）用到的程序，全在https://github.com/dlunion/CCDL/tree/master/caffe-easy 这个版本的caffe里面，该caffe主要运行在windows下，可以复制里面主要的层和程序也可以完成任务，也可以下载编译好的程序http://www.zifuture.com/fs/12.github/mtcnn/caffe-build-cuda8.0.rar 来训练，里面有提供<br/>

<br/>

## 有关键点的训练
产生一个label.txt的时候，格式是:<br/>
   samples/filename.jpg xmin ymin xmax ymax ptsx ptsy ptsx ptsy<br/>
然后相对应的修改py代码(gen_24net_list.py、gen_48net_list.py)里面的has_pts为True，注意里面-1的个数要跟你的pts个数对上（尴尬没写傻瓜式一点）<br/>

<br/>

## 一个图多个box的训练
主要是box的处理在gen_12net_data2.py的bbox = map(float, annotation[1:5])部分，这里限制了只读取1个box，如果多个box可以修改5为-1，当然这时候如果你又有pts就得自己修改啦。<br/>

<br/>

## 从只有图片开始
如果你只有图片，然后开始训练，那么可以利用目录下的[lab.py](https://github.com/dlunion/CCDL/tree/master/tools/ssd-lab)，用来制作目标框，当得到相对应的目标框文件后，可以执行gen-st-ann.py这个py程序转换为上面说的label.txt所需要的样子

## 训练程序
如果你装好了cuda8.0，那么可以直接下载exe程序，立马就可以训练，而不需要自己编译<br/>
[caffe-build-cuda8.0.rar](http://www.zifuture.com/fs/12.github/mtcnn/caffe-build-cuda8.0.rar) <br/>
[caffe-buildx64-cpu.rar](http://www.zifuture.com/fs/12.github/mtcnn/caffe-buildx64-cpu.rar)
