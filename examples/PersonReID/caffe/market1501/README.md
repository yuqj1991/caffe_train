

## Preparation
- download Market-1501 dataset and put Market-1501 in $HOME/datasets/
- `cd examples/market1501/mat-codes` and `matlab -nodisplay -r  'generate_train(), exit()'` to generate train, test and qurey data lists.
- Build with NCLL / cuda-8.0 / cudnn-v5.1

## Results on Market-1501

[Market-1501](http://liangzheng.com.cn/Project/state_of_the_art_market1501.html) is one of the most popular person re-identification datasets.

Models can be found in `models/market1501/model_name`

Many scripts (e.g initialization, testing, training, extract feature and evaluation) can be found in `examples/market1501/`

[iter_size * batch_size] = real batch_size

### CaffeNet
- Link to the [pre-trained CaffeNet model](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel)
- `python models/market1501/generate_caffenet.py` for generate caffenet based person re-ID network and solver files.
- `sh models/market1501/caffenet/train.sh --gpu 0` for training models.
- `sh examples/market1501/extract/extract_prediction.sh 0 caffenet fc7` for extracting features of query and test data
- `cd examples/market1501/evaluation/` and `evaluation('caffenet')` to evaluate performance of the trained model on Market-1501
- final results are [1x128] : mAP = 0.402689, r1 precision = 0.639846 [Euclidean]

### GoogleNet
- Link to the [pre-trained GoogleNet model](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)
- GoogleNet-v1 model is already in `models/market1501/googlenet`
- `sh models/market1501/googlenet/train.sh --gpu 0`
- `sh examples/market1501/extract/extract_prediction.sh 0 googlenet pool5/7x7_s1`
- `cd examples/market1501/evaluation/` and `evaluation('googlenet')`
- final results are : mAP = 0.511545, r1 precision = 0.735154 [Cos + Eucl]

### VGG-16
- Link to the [pre-trained VGG-16 model](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel)
- `python models/market1501/generate_vgg16.py` for generate caffenet based person re-ID network and solver files.
- `sh models/market1501/vgg16/train.sh --gpu 2,3` for training
- `sh examples/market1501/extract/extract_prediction.sh 0 vgg16 fc7` for extracting features
- `cd examples/market1501/evaluation/` and `evaluation('vgg16')` to evaluate performance of vgg16/fc7 on Market-1501
- final results are [2x 24] : mAP = 0.456417, r1 precision = 0.677257

### resnet-50
- `python models/market1501/generate_resnet50.py`
- `sh models/market1501/res50/train.sh --gpu 2,3`
- `sh examples/market1501/extract/extract_prediction.sh 0 res50 pool5`
- final results are : mAP = 0.585765, r1 precision = 0.790974 [Cos + Eucl]
