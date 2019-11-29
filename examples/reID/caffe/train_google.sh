#!/usr/bin/env sh
model_dir=models/market1501/googlenet
pre_train_dir=${HOME}/datasets/model_pretrained/googlenet

../../../build/tools/caffe train --solver ./prototxt/googlenet/solver.prototxt -gpu 0 \
#--snapshot=../snapshot/reid_iter_6824.solverstate
