#!/usr/bin/env sh
# Create the lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=examples/market1501

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

echo "Creating train lmdb..."

rm -rf $EXAMPLE/market1501_train_lmdb

GLOG_logtostderr=1 build/$EXAMPLE/convert_market1501.bin \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --check_size=1 \
    $EXAMPLE/lists/train.lst \
    $EXAMPLE/market1501_train_lmdb

echo "Creat mean proto"

sh $EXAMPLE/shell/make_market1501_mean.sh

echo "Remove train lmdb..."
rm -rf $EXAMPLE/market1501_train_lmdb
