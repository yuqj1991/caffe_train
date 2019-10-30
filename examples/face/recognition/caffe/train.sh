#!/bin/sh
if ! test -f ./softmax_loss/inception_tiny/inception_resnet_v2_tiny_train.prototxt ;then
	echo "error: inception_resnet_v2_tiny_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../../build/tools/caffe train --solver=./softmax_loss/inception_tiny/inception_resnet_v2_tiny_solver.prototxt -gpu 2 \
#--snapshot=../snapshot/face_iter_5000.solverstate
