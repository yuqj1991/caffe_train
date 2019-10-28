#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
python models/market1501/generate_caffenet.py
python models/market1501/generate_vgg16.py
python models/market1501/generate_resnet50.py
