#!/usr/bin/env sh
# Compute the mean image from the market1501 training lmdb

EXAMPLE=examples/market1501
TOOLS=build/tools

GLOG_logtostderr=1 $TOOLS/compute_image_mean $EXAMPLE/market1501_train_lmdb
#  $EXAMPLE/market1501_mean.binaryproto

echo "Done."
