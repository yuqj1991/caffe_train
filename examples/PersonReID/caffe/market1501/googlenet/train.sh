#!/usr/bin/env sh
model_dir=models/market1501/googlenet
pre_train_dir=${HOME}/datasets/model_pretrained/googlenet

GLOG_log_dir=${model_dir}/log ./build/tools/caffe train --solver ${model_dir}/solver.proto --weights ${pre_train_dir}/bvlc_googlenet.caffemodel $@
