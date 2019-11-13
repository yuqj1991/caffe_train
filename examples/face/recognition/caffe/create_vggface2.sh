#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=../../../../../dataset/facedata/recognition
DATA=.
TOOLS=../../../../build/tools

TRAIN_DATA_ROOT=../../../../../dataset/facedata/recognition/vggface/vggface2_align_train/
TEST_DATA_ROOT=../../../../../dataset/facedata/recognition/vggface/vggface2_align_test/

LABEL_TRAIN_FILE=vggface2_train.txt
LABEL_TEST_FILE=vggface2_test.txt

TRAIN_LMDB=face_recog_vggface2_lmdb_train
TEST_LMDB=face_recog_vggface2_lmdb_test

# Set RESIZE=true to resize the images to 128x128. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=128
  RESIZE_WIDTH=128
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

#if [ ! -d "$VAL_DATA_ROOT" ]; then
#  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
#  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet validation data is stored."
#  exit 1 vggface2_align_train face_recog_vggface2_lmdb
#fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --encode_type=jpg \
    --encoded=true \
    $TRAIN_DATA_ROOT \
    $DATA/$LABEL_TRAIN_FILE \
    $EXAMPLE/$TRAIN_LMDB

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --encode_type=jpg \
    --encoded=true \
    $TEST_DATA_ROOT \
    $DATA/$LABEL_TEST_FILE \
    $EXAMPLE/$TEST_LMDB

echo "Done."
